---
layout: product_landing
title: Support · Lume
description: "Lume support — contact, download, trial, licensing, and common questions."
permalink: /lume/support/
product_theme: lume
product_css: /assets/css/lume.css
product_icon: /assets/images/lume/favicon-32.png
---

{% assign dl = "https://downloads.machygienist.app/releases/Lume-latest.zip" %}
{% include lume-header.html %}

<main class="lume-legal lume-wrap">
  <h1>Support</h1>
  <p class="lume-legal-lede">
    Help with install, playback, trial, or licensing.
  </p>

  <section class="lume-support-card">
    <h2>Contact</h2>
    <p>
      Email
      <a href="mailto:lume@zlu.me?subject=Lume%20Support">lume@zlu.me</a>.
      Include your macOS version, Lume version (About), and what you tried. We usually reply within a few business days.
    </p>
  </section>

  <section class="lume-support-card">
    <h2>Download</h2>
    <p>
      Apple Silicon · macOS 14+.
      <a href="{{ dl }}">Download the latest build</a>.
    </p>
  </section>

  <section>
    <h2>Common questions</h2>

    <div class="lume-faq-item">
      <h3>What platforms are supported?</h3>
      <p>Apple Silicon Macs on macOS 14 or later. Intel Macs are not supported.</p>
    </div>

    <div class="lume-faq-item">
      <h3>How does the trial work?</h3>
      <p>
        Full features for seven days from first launch. After that, buy a one-time license ($19 via Polar) to keep
        playing. No subscription.
      </p>
    </div>

    <div class="lume-faq-item">
      <h3>How do I activate a license?</h3>
      <p>
        Purchase through Polar checkout, then paste the license key in Lume (License / Activate). Activation uses
        Polar’s customer portal API on your Mac.
      </p>
    </div>

    <div class="lume-faq-item">
      <h3>Can Lume open YouTube or similar pages?</h3>
      <p>
        Paste the page URL when a bundled or installed yt-dlp helper can resolve it. Site availability changes; if a
        page fails, try a direct media URL or a local file.
      </p>
    </div>

    <div class="lume-faq-item">
      <h3>Do you upload my videos?</h3>
      <p>
        No. Local playback stays on your Mac. Optional Live ASR sends short audio slices only when you enable it and
        configure a key. See
        <a href="/lume/privacy/">Privacy</a>.
      </p>
    </div>

    <div class="lume-faq-item">
      <h3>Where are legal pages?</h3>
      <p>
        <a href="/lume/privacy/">Privacy</a>
        ·
        <a href="/lume/terms/">Terms</a>
      </p>
    </div>
  </section>

  <p class="lume-legal-meta">
    Still stuck?
    <a href="mailto:lume@zlu.me?subject=Lume%20Support">lume@zlu.me</a>
  </p>
</main>

{% include lume-footer.html %}
