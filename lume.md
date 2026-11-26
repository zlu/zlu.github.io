---
layout: product_landing
title: Lume
description: "Mac cinema player for local files and network streams. VLCKit codecs, yt-dlp URLs, A-B loops, live captions. 7-day trial, then $19."
keywords: "Lume, macOS video player, VLCKit, yt-dlp, cinema player, A-B loop, live captions, network stream"
permalink: /lume/
product_theme: lume
product_css: /assets/css/lume.css
product_icon: /assets/images/lume/favicon-32.png
image: /assets/images/lume/screenshot.png
---

{% assign dl = "https://downloads.machygienist.app/releases/Lume-latest.zip" %}

{% include lume-header.html %}

<main>
  <section class="lume-hero lume-wrap">
    <div class="lume-hero-copy">
      <h1>Lume</h1>
      <p class="lume-hero-kicker">Cinema chrome. Real codecs.</p>
      <p class="lume-hero-lede">
        A Mac player for local files and network streams. Open a folder, drop a file, or paste a URL. Built on VLCKit.
      </p>
      <div class="lume-cta-row">
        <a class="lume-btn lume-btn-primary" href="{{ dl }}">Download for Mac</a>
        <a class="lume-btn lume-btn-ghost" href="#features">Features</a>
      </div>
      <p class="lume-meta">Apple Silicon · macOS 14+ · 7-day trial · $19 one-time</p>
    </div>
    <div class="lume-hero-visual">
      <img
        class="lume-shot"
        src="/assets/images/lume/screenshot.png"
        width="1280"
        height="800"
        alt="Lume playing a local MP4 with transport controls">
    </div>
  </section>

  <section class="lume-features lume-wrap" id="features">
    <p class="lume-features-label">Features</p>
    <ul class="lume-feature-grid">
      <li class="lume-feature">
        <h3>Local + queue</h3>
        <p>Multi-select open, drag-and-drop, playlist advance, and continue watching on launch.</p>
      </li>
      <li class="lume-feature">
        <h3>Network URL</h3>
        <p>Paste HTTP streams or site pages. yt-dlp resolves YouTube and friends when available.</p>
      </li>
      <li class="lume-feature">
        <h3>VLCKit codecs</h3>
        <p>The formats you already expect from VLC, in a native SwiftUI shell.</p>
      </li>
      <li class="lume-feature">
        <h3>Scrub preview</h3>
        <p>Seek with thumbnail previews for local files via ffmpeg helpers.</p>
      </li>
      <li class="lume-feature">
        <h3>Captions</h3>
        <p>Sidecar SRT/VTT, embedded tracks, delay nudge, and optional live ASR.</p>
      </li>
      <li class="lume-feature">
        <h3>A-B loops</h3>
        <p>Mark loops per video, save them beside the file, jump back from the drawer.</p>
      </li>
      <li class="lume-feature">
        <h3>Cinema chrome</h3>
        <p>Autohide controls, translucent drawer, picture adjust, aspect and rotate.</p>
      </li>
      <li class="lume-feature">
        <h3>PiP + pin</h3>
        <p>Floating picture-in-picture sizes, always-on-top, fullscreen, keyboard map.</p>
      </li>
      <li class="lume-feature">
        <h3>One-time license</h3>
        <p>Full trial for seven days. Then $19 via Polar. No subscription.</p>
      </li>
    </ul>
  </section>
</main>

{% include lume-footer.html %}
