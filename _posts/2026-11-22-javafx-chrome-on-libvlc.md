---
layout: post
title: "Same JavaFX chrome, libVLC underneath"
description: "I kept the mini-player UI from the JavaFX Media post and swapped the engine for libVLC via vlcj, including how video reaches an ImageView on macOS and what bundling natives implies."
keywords: "vlcj, libVLC, JavaFX, ImageViewVideoSurface, PixelBuffer, desktop player, LGPL"
date: 2026-11-22
comments: true
tags:
  - VLC
  - libVLC
  - vlcj
  - JavaFX
  - desktop
---

The [JavaFX Media mini-player](/2026/11/22/javafx-media-local-mp4/) was enough to learn the basic UI components of JavaFX. The codec engine is quite weak. In this post, I'll replace the native JavaFX media player with libVLC (through [vlcj](https://github.com/caprica/vlcj)), which is like swapping out the engine of a car for a more capable one.

Code lives in [github.com/zlu/javafx-mini-player](https://github.com/zlu/javafx-mini-player). `mvn javafx:run` now starts `VlcMiniPlayerApp`. The original Media-only class is still in the tree if you pass `-Djavafx.mainClass=io.github.zlu.javafxplayer.MiniPlayerApp`.

```sh
git clone https://github.com/zlu/javafx-mini-player.git
cd javafx-mini-player
mvn javafx:run -Djavafx.args="$PWD/sample-h264.mp4"
```

One extra requirement: a system [VLC](https://www.videolan.org/vlc/#download) install that exposes `libvlc`. On my Mac, `/Applications/VLC.app` is enough; vlcj's native discovery finds the dylibs without me setting `jna.library.path` by hand.

## How VLC is stacked

VLC is a thin UI on top of a C library stack. The piece you embed is **libVLC**. Under that sits **libVLCcore**, which owns the clock, loads modules, and runs the pipeline (access → demux → decode → filters → audio/video out). Many codecs arrive through FFmpeg's `avcodec`; VLC also ships its own demuxers and protocol modules. The familiar Qt UI on Windows/Linux (and the native UI module on macOS) talks to the same libVLC surface that a Java binding uses.

```text
Qt / macOS UI          JavaFX chrome (this post)
        \                    /
         \                  /
          └── libVLC  ← vlcj binds here
                 │
            libVLCcore
                 │
         modules (+ often FFmpeg)
```

## Dependencies

I added [vlcj](https://github.com/caprica/vlcj) for the libVLC API and [vlcj-javafx](https://github.com/caprica/vlcj-javafx) for the JavaFX video surface. `javafx-controls` stays for buttons and layout. I left `javafx-media` in the POM so the Part 1 sample still compiles; a libVLC-only app would not need it.

```xml
<dependency>
  <groupId>uk.co.caprica</groupId>
  <artifactId>vlcj</artifactId>
  <version>4.12.1</version>
</dependency>
<dependency>
  <groupId>uk.co.caprica</groupId>
  <artifactId>vlcj-javafx</artifactId>
  <version>1.2.1</version>
</dependency>
```

Pin both together. `vlcj-javafx` 1.2.1 expects vlcj 4.12.x. If you pair an older `vlcj-javafx` 1.2.0 with a newer 4.11+ core, you can hit an `AbstractMethodError` on `newFormatSize`: audio still plays, video stays black, and the failure mode looks like a codec bug until you read the stack trace.

## Getting frames into JavaFX

With libVLC, we are forced to render video directly into an `ImageView`: libVLC fills a native pixel buffer, and your toolkit is responsible for putting those pixels on screen.

`ImageViewVideoSurface` does the JavaFX side. VLC writes BGRA into a `PixelBuffer`; you wrap that in a `WritableImage` and show it on an `ImageView`. It costs more CPU than a native embed, but it is the approach that actually runs on macOS, Windows, and Linux with the same code.

```java
MediaPlayerFactory factory = new MediaPlayerFactory();
EmbeddedMediaPlayer mediaPlayer =
        factory.mediaPlayers().newEmbeddedMediaPlayer();

ImageView videoView = new ImageView();
videoView.setPreserveRatio(true);
mediaPlayer.videoSurface().set(new ImageViewVideoSurface(videoView));
```

I create the factory and player in `Application.init()`, attach the surface before the first `play`, and size the `ImageView` by binding `fitWidth` / `fitHeight` to the center pane the same way Part 1 bound `MediaView`.

## Wiring UI with libVLC

The window is still a `BorderPane`: Open + status on top, video in the center, Play/Pause + seek + time on the bottom. What changes is the call you make when those controls fire.

Opening a file used to mean constructing `Media` + `MediaPlayer` and waiting for `setOnReady`. With vlcj you hand a path (or MRL) to `media().play(...)` and return. Playback starts on VLC's threads; length and current time show up later as events.

```java
private void load(File file) {
    status.setText(file.getName());
    boolean ok = mediaPlayer.media().play(file.getAbsolutePath());
    if (!ok) {
        status.setText("Could not start: " + file.getName());
    }
}
```

I register a `MediaPlayerEventAdapter` once at startup. `lengthChanged` enables the slider and sets its max in milliseconds. `timeChanged` pushes the playhead into the slider and the time label, unless the user is mid-drag. Those callbacks are **not** on the JavaFX Application Thread, so every UI touch goes through `Platform.runLater` (same rule as any background work in Part 1).

```java
mediaPlayer.events().addMediaPlayerEventListener(new MediaPlayerEventAdapter() {
    @Override
    public void lengthChanged(MediaPlayer player, long newLength) {
        Platform.runLater(() -> {
            lengthMs = Math.max(0, newLength);
            seek.setMax(lengthMs > 0 ? lengthMs : 100);
            seek.setDisable(false);
            playPause.setDisable(false);
        });
    }

    @Override
    public void timeChanged(MediaPlayer player, long newTime) {
        Platform.runLater(() -> {
            if (!seeking.get()) {
                seek.setValue(newTime);
            }
            timeLabel.setText(format(newTime) + " / " + format(lengthMs));
        });
    }
});
```

Play/Pause becomes `controls().play()` / `controls().pause()`, keyed off `status().isPlaying()`. Seek is `controls().setTime(ms)`. The drag flag from Part 1 still matters: while `valueChangingProperty` is true I stop copying clock updates into the thumb, and when the drag ends I seek to the slider value. Without that, VLC's time ticks yank the thumb back to the playhead for the whole gesture.

```java
seek.valueChangingProperty().addListener((obs, was, changing) -> {
    seeking.set(changing);
    if (!changing) {
        mediaPlayer.controls().setTime((long) seek.getValue());
    }
});
```

For a quick map if you are porting call sites from Part 1:

| UI action | JavaFX Media | libVLC via vlcj |
| --- | --- | --- |
| Open file | `new MediaPlayer(new Media(uri))` | `mediaPlayer.media().play(path)` |
| Play / pause | `play()` / `pause()` | `controls().play()` / `controls().pause()` |
| Seek | `seek(Duration.millis(...))` | `controls().setTime(ms)` |
| Clock → slider | `currentTimeProperty()` | `timeChanged` |
| Duration ready | `setOnReady` + `getTotalDuration()` | `lengthChanged` |

## Releasing native state

Nulling the Java field does not free libVLC. On window close and in `Application.stop` I stop playback, then `release()` the player and the factory, once. A second release is a reliable way to take the JVM down in native code.

```java
mediaPlayer.controls().stop();
mediaPlayer.release();
factory.release();
```

## Shipping, briefly

The sample expects the user to already have VLC installed. I don't want to bundle multi-hundred-megabyte native trees in git.

Shipping libVLC with your application requires LGPL compliance.