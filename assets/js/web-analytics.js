/**
 * First-party web funnel for zlu.me → BetterReads admin (?app=zlume).
 * Ingest: PUT https://www.betterreads.dev/api/zlume-web-stats
 */
(function () {
  "use strict";

  var STORAGE_KEY = "zlume_web_analytics_v1";
  var SESSION_KEY = "zlume_web_session_v1";
  var OWNER_EMAIL_KEY = "zlume_analytics_email_v1";
  var OWNER_CLAIM_KEY = "zlume_analytics_owner_v1";
  var MAX_EVENTS = 80;
  var USAGE_TICK_MS = 15000;
  var USAGE_FLUSH_MS = 60000;
  // GitHub Pages has no server — ingest on betterreads-fe (same host as /admin).
  // Never use a relative /api/... URL: jekyll serve (Webrick) rejects PUT.
  var INGEST_URL = "https://www.betterreads.dev/api/zlume-web-stats";

  var INTERNAL_EMAILS = {
    "zlututubj@qq.com": true,
    "zlu@me.com": true,
  };

  function randomId() {
    if (typeof crypto !== "undefined" && crypto.randomUUID) {
      return crypto.randomUUID();
    }
    return Date.now().toString(36) + "-" + Math.random().toString(36).slice(2);
  }

  function readState() {
    try {
      var saved = localStorage.getItem(STORAGE_KEY);
      if (saved) return JSON.parse(saved);
    } catch (e) {
      /* ignore */
    }
    return null;
  }

  function persistLocal(state) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
      return true;
    } catch (e) {
      return false;
    }
  }

  function isInternalEmail(email) {
    if (!email) return false;
    return !!INTERNAL_EMAILS[String(email).trim().toLowerCase()];
  }

  function hasOwnerClaim() {
    try {
      return localStorage.getItem(OWNER_CLAIM_KEY) === "1";
    } catch (e) {
      return false;
    }
  }

  function inferAcquisition(params) {
    var utmSource = params.get("utm_source");
    var utmMedium = params.get("utm_medium");
    var utmCampaign = params.get("utm_campaign");
    if (utmSource) {
      return { source: utmSource, medium: utmMedium, campaign: utmCampaign };
    }
    if (params.has("fbclid")) {
      return {
        source: "facebook",
        medium: utmMedium || "paid_social",
        campaign: utmCampaign,
      };
    }
    if (params.has("gclid") || params.has("wbraid") || params.has("gbraid")) {
      return {
        source: "google",
        medium: utmMedium || "cpc",
        campaign: utmCampaign,
      };
    }
    return { source: null, medium: utmMedium, campaign: utmCampaign };
  }

  function initState(path, params) {
    var now = new Date().toISOString();
    var acq = inferAcquisition(params);
    return {
      visitorId: randomId(),
      firstSeenAt: now,
      lastSeenAt: now,
      sessionCount: 0,
      pageViewCount: 0,
      entryPath: path,
      referrer: document.referrer || null,
      source: acq.source,
      medium: acq.medium,
      campaign: acq.campaign,
      totalUsageSeconds: 0,
      featureUsage: {},
      funnelReached: [],
      funnelEvents: [],
    };
  }

  function applyAcquisitionIfMissing(state, params) {
    if (state.source) return;
    var acq = inferAcquisition(params);
    if (acq.source) {
      state.source = acq.source;
      state.medium = state.medium || acq.medium;
      state.campaign = state.campaign || acq.campaign;
    }
  }

  function normalizePath(pathname) {
    if (!pathname) return "/";
    if (pathname.length > 1 && pathname.charAt(pathname.length - 1) === "/") {
      return pathname;
    }
    // Jekyll pretty URLs often end with /; .html posts do not.
    return pathname;
  }

  function milestoneForPath(pathname) {
    var p = normalizePath(pathname);
    if (p === "/" || p === "/index.html") return "home_viewed";
    if (p === "/posts" || p === "/posts/" || p === "/posts/index.html") {
      return "posts_opened";
    }
    if (p === "/teach" || p === "/teach/" || p === "/teach/index.html") {
      return "teach_opened";
    }
    if (p.indexOf("/teach/courses/") === 0) return "teach_course_opened";
    if (p.indexOf("/teach/universities/") === 0) {
      return "teach_university_opened";
    }
    if (p.indexOf("/teach/") === 0) return "teach_opened";
    if (
      p === "/portfolio" ||
      p === "/portfolio/" ||
      p === "/portfolio/index.html"
    ) {
      return "portfolio_opened";
    }
    if (
      p === "/contact" ||
      p === "/contact/" ||
      p === "/contact/index.html" ||
      p === "/contact.html"
    ) {
      return "contact_opened";
    }
    if (
      /^\/20\d{2}\//.test(p) ||
      p.indexOf("/ai-glossary/") === 0 ||
      p.indexOf("/med-glossary/") === 0 ||
      p.indexOf("/cs-glossary/") === 0 ||
      p.indexOf("/algos/") === 0
    ) {
      return "post_opened";
    }
    return null;
  }

  function featureForPath(pathname) {
    var p = normalizePath(pathname);
    if (p === "/" || p === "/index.html") return "home";
    if (p.indexOf("/teach/courses/") === 0) return "teach_course";
    if (p.indexOf("/teach/universities/") === 0) return "teach_university";
    if (p.indexOf("/teach") === 0) return "teach";
    if (p.indexOf("/posts") === 0) return "posts";
    if (p.indexOf("/portfolio") === 0) return "portfolio";
    if (p.indexOf("/contact") === 0) return "contact";
    if (/^\/20\d{2}\//.test(p)) return "post";
    if (p.indexOf("/ai-glossary/") === 0) return "ai_glossary";
    if (p.indexOf("/med-glossary/") === 0) return "med_glossary";
    if (p.indexOf("/cs-glossary/") === 0) return "cs_glossary";
    if (p.indexOf("/algos/") === 0) return "algos";
    return null;
  }

  function appendMilestone(state, name, path, at) {
    state.funnelEvents.push({ name: name, path: path, at: at });
    var reached = state.funnelReached || [];
    if (reached.indexOf(name) === -1) reached = reached.concat([name]);
    state.funnelReached = reached;
    if (state.funnelEvents.length > MAX_EVENTS) {
      state.funnelEvents = state.funnelEvents.slice(-MAX_EVENTS);
    }
  }

  function incrementFeatureOnState(state, feature, path) {
    var key = String(feature || "")
      .trim()
      .slice(0, 48);
    if (!key) return;
    var usage = Object.assign({}, state.featureUsage || {});
    usage[key] = (usage[key] || 0) + 1;
    state.featureUsage = usage;
    var eventName = "feature_" + key;
    if ((state.funnelReached || []).indexOf(eventName) === -1) {
      appendMilestone(state, eventName, path, new Date().toISOString());
    }
  }

  function shouldSendRemote() {
    var host = window.location.hostname;
    // Skip jekyll serve / previews so Webrick is not hit with PUT and
    // local traffic does not pollute production analytics.
    return host === "zlu.me" || host === "www.zlu.me";
  }

  function persistAndSend(state) {
    if (!persistLocal(state)) return;
    if (!shouldSendRemote()) return;

    var email;
    try {
      email = sessionStorage.getItem(OWNER_EMAIL_KEY) || undefined;
    } catch (e) {
      /* ignore */
    }
    var ownerClaim = hasOwnerClaim() || isInternalEmail(email);

    var body = Object.assign({}, state, {
      platform: "web",
      language: navigator.language,
      screenWidth: window.screen.width,
      screenHeight: window.screen.height,
      timezone:
        (Intl.DateTimeFormat().resolvedOptions().timeZone || null),
      timezoneOffsetMinutes: -new Date().getTimezoneOffset(),
      totalUsageSeconds: state.totalUsageSeconds || 0,
      featureUsage: state.featureUsage || {},
    });
    if (email) body.email = email;
    if (ownerClaim) body.ownerClaim = true;

    try {
      fetch(INGEST_URL, {
        method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
        keepalive: true,
        mode: "cors",
      }).catch(function () {});
    } catch (e) {
      /* ignore */
    }
  }

  function flushWebAnalytics() {
    var state = readState();
    if (state) persistAndSend(state);
  }

  function trackFunnelMilestone(name, pathOverride) {
    var path = pathOverride || window.location.pathname;
    var params = new URLSearchParams(window.location.search);
    var location = params.toString() ? path + "?" + params : path;
    var state = readState() || initState(location, params);
    applyAcquisitionIfMissing(state, params);
    var at = new Date().toISOString();
    state.lastSeenAt = at;
    if ((state.funnelReached || []).indexOf(name) === -1) {
      appendMilestone(state, name, path, at);
    }
    persistAndSend(state);
  }

  function addVisibleUsageSeconds(seconds, pathname) {
    if (!Number.isFinite(seconds) || seconds <= 0) return;
    var state = readState();
    if (!state) return;
    state.totalUsageSeconds = (state.totalUsageSeconds || 0) + seconds;
    state.lastSeenAt = new Date().toISOString();
    if ((state.totalUsageSeconds || 0) >= 300) {
      if ((state.funnelReached || []).indexOf("used_5min") === -1) {
        appendMilestone(
          state,
          "used_5min",
          pathname || window.location.pathname,
          state.lastSeenAt,
        );
      }
    }
    persistLocal(state);
  }

  function recordPageView(pathname, searchParams) {
    var params =
      searchParams || new URLSearchParams(window.location.search);
    var location = params.toString()
      ? pathname + "?" + params
      : pathname;
    var state = readState() || initState(location, params);
    applyAcquisitionIfMissing(state, params);

    var now = new Date().toISOString();
    state.lastSeenAt = now;
    state.pageViewCount = (state.pageViewCount || 0) + 1;

    var sessionId = null;
    try {
      sessionId = sessionStorage.getItem(SESSION_KEY);
    } catch (e) {
      /* ignore */
    }
    if (!sessionId) {
      sessionId = randomId();
      try {
        sessionStorage.setItem(SESSION_KEY, sessionId);
      } catch (e) {
        /* ignore */
      }
      state.sessionCount = (state.sessionCount || 0) + 1;
    }

    var milestone = milestoneForPath(pathname);
    if (milestone && (state.funnelReached || []).indexOf(milestone) === -1) {
      appendMilestone(state, milestone, pathname, now);
    }

    var feature = featureForPath(pathname);
    if (feature) incrementFeatureOnState(state, feature, pathname);

    appendMilestone(state, "page_view", pathname, now);
    persistAndSend(state);
  }

  function startUsageTicker() {
    var lastTick = Date.now();
    var lastFlush = 0;

    function credit(forceSend) {
      var now = Date.now();
      var elapsed = now - lastTick;
      lastTick = now;
      if (
        document.visibilityState === "visible" &&
        elapsed > 0 &&
        elapsed < 120000
      ) {
        addVisibleUsageSeconds(elapsed / 1000, window.location.pathname);
      }
      if (forceSend || now - lastFlush >= USAGE_FLUSH_MS) {
        lastFlush = now;
        flushWebAnalytics();
      }
    }

    document.addEventListener("visibilitychange", function () {
      if (document.visibilityState === "hidden") credit(true);
      else lastTick = Date.now();
    });
    window.addEventListener("pagehide", function () {
      credit(true);
    });
    window.setInterval(function () {
      if (document.visibilityState === "visible") credit(false);
    }, USAGE_TICK_MS);
  }

  function wireCtaClicks() {
    document.addEventListener(
      "click",
      function (event) {
        var el = event.target;
        if (!el || !el.closest) return;
        var scrollBook = el.closest("[data-teach-scroll-book]");
        if (scrollBook) {
          trackFunnelMilestone("wechat_cta_clicked");
          return;
        }
        var anchor = el.closest("a");
        if (!anchor) return;
        var href = anchor.getAttribute("href") || "";
        if (/^mailto:hello@zlu\.me/i.test(href)) {
          trackFunnelMilestone("email_cta_clicked");
          return;
        }
        if (/matheion\.zlu\.me/i.test(href)) {
          trackFunnelMilestone("matheion_cta_clicked");
        }
      },
      true,
    );
  }

  function boot() {
    var pathname = window.location.pathname || "/";
    var params = new URLSearchParams(window.location.search);
    recordPageView(pathname, params);
    startUsageTicker();
    wireCtaClicks();
  }

  window.ZlumeWebAnalytics = {
    trackFunnelMilestone: trackFunnelMilestone,
    flush: flushWebAnalytics,
    recordPageView: recordPageView,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
