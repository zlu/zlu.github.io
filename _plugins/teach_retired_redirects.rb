# frozen_string_literal: true

module Jekyll
  # Soft-redirect retired bulk university shells (cut from the directory farm)
  # to the evidence-backed universities index. Pages are omitted from the sitemap.
  class TeachRetiredRedirects < Generator
    safe true
    priority :low

    TARGET = "/teach/universities/"

    def generate(site)
      # Opt-in: set teach_retired_redirects: true in _config.yml (adds ~3.8k soft redirects).
      return unless site.config["teach_retired_redirects"]

      raw = site.data["teach_retired_slugs"]
      slugs = if raw.is_a?(Hash)
                Array(raw["slugs"])
              elsif raw.is_a?(Array)
                raw
              else
                []
              end
      return if slugs.empty?

      # Skip live uni pages and any explicit redirect already in the site.
      occupied = {}
      Array(site.data["teach_universities"]).each do |u|
        next unless u.is_a?(Hash)

        occupied[u["slug"].to_s] = true
      end
      site.pages.each do |page|
        url = page.url.to_s
        next unless url.start_with?("/teach/universities/") && url.count("/") == 3

        occupied[url.delete_prefix("/teach/universities/").delete_suffix("/")] = true
      end

      count = 0
      slugs.each do |slug|
        slug = slug.to_s.strip
        next if slug.empty? || occupied[slug]

        page = TeachRetiredRedirectPage.new(site, site.source, slug)
        site.pages << page
        count += 1
      end

      Jekyll.logger.info "TeachRetired:", "redirected #{count} retired university shells → #{TARGET}"
    end
  end

  class TeachRetiredRedirectPage < Page
    def initialize(site, base, slug)
      @site = site
      @base = base
      @dir = "teach/universities/#{slug}"
      @name = "index.html"
      process(@name)
      self.data = {
        "layout" => "redirect",
        "redirect_to" => "/teach/universities/",
        "sitemap" => false,
        "title" => "Redirecting…",
        "permalink" => "/teach/universities/#{slug}/"
      }
      self.content = ""
    end
  end
end
