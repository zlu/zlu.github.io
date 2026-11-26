---
layout: product_landing
title: Privacy · Lume
description: "How Lume handles local playback, network streams, licensing, and optional live captions."
permalink: /lume/privacy/
product_theme: lume
product_css: /assets/css/lume.css
product_icon: /assets/images/lume/favicon-32.png
---

{% include lume-header.html %}

<main class="lume-legal lume-wrap">
  <h1>Privacy</h1>
  <p class="lume-legal-lede">
    Lume is a native macOS video player. Playback and most preferences stay on your Mac.
  </p>
  <p class="lume-legal-meta">Last updated: November 26, 2026</p>

  <section>
    <h2>Local playback</h2>
    <p>
      Local files, playlists, continue-watching state, A-B loops, and app preferences are stored on your device
      (for example under Application Support). We do not upload your media library or file contents to our servers.
    </p>
  </section>

  <section>
    <h2>Network streams</h2>
    <p>
      When you paste a URL or open a network stream, Lume contacts that host (and may use bundled helpers such as
      yt-dlp) to resolve and play media. Those requests follow the third party’s terms and privacy policy.
      We do not operate the remote sites you choose to open.
    </p>
  </section>

  <section>
    <h2>Optional live captions</h2>
    <p>
      Live ASR is optional. When you enable it and provide an API key, short audio slices leave your Mac for the
      speech service you configured (currently DeepSeek ASR). Caption text returned by that service is shown in the
      player. Turn Live ASR off, or omit the key, to keep audio on-device.
    </p>
  </section>

  <section>
    <h2>Licensing</h2>
    <p>
      Trial entitlement is tracked locally. Paid license activation and validation contact Polar (merchant of record)
      with your license key and machine identifiers needed for activation limits. Media files and playback history
      are not included in those requests.
    </p>
  </section>

  <section>
    <h2>Updates</h2>
    <p>
      The app may check a download host for newer builds (appcast / release ZIP). That check does not send your
      library or playback history.
    </p>
  </section>

  <section>
    <h2>What we do not collect</h2>
    <ul>
      <li>No account required to try or use the player</li>
      <li>No advertising identifiers or third-party ad trackers in the app</li>
      <li>No sale of personal data</li>
    </ul>
  </section>

  <section>
    <h2>Terms</h2>
    <p>
      License, warranty, and liability limits are in our
      <a href="/lume/terms/">Terms</a>.
    </p>
  </section>

  <section>
    <h2>Contact</h2>
    <p>
      Questions about this policy or Lume:
      <a href="mailto:lume@zlu.me">lume@zlu.me</a>.
    </p>
  </section>
</main>

{% include lume-footer.html %}
