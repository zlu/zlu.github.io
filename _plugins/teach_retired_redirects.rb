# frozen_string_literal: true

module Jekyll
  # Soft-redirect retired bulk university shells (cut from the directory farm)
  # to the evidence-backed universities index. Pages are omitted from the sitemap.
  #
  # Emits redirects for the bare uni slug and each service suffix
  # (preview / assignment / exam / sync) so GSC corpse URLs stop 404-ranking.
  class TeachRetiredRedirects < Generator
    safe true
    priority :low

    TARGET = "/teach/universities/"
    SERVICES = %w[preview assignment exam sync].freeze

    def generate(site)
      # Opt-in: set teach_retired_redirects: true in _config.yml
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

      occupied = occupied_teach_uni_paths(site)

      count = 0
      slugs.each do |slug|
        slug = slug.to_s.strip
        next if slug.empty?

        paths_for(slug).each do |path|
          next if occupied[path]

          site.pages << TeachRetiredRedirectPage.new(site, site.source, path)
          occupied[path] = true
          count += 1
        end
      end

      Jekyll.logger.info "TeachRetired:", "redirected #{count} retired university paths → #{TARGET}"
    end

    def occupied_teach_uni_paths(site)
      occupied = {}

      Array(site.data["teach_universities"]).each do |u|
        next unless u.is_a?(Hash)

        slug = u["slug"].to_s
        next if slug.empty?

        paths_for(slug).each { |path| occupied[path] = true }
      end

      site.pages.each do |page|
        url = page.url.to_s
        next unless url.start_with?("/teach/universities/")

        path = url.delete_prefix("/teach/universities/").delete_suffix("/")
        next if path.empty?

        parts = path.split("/")
        next unless parts.length == 1 || (parts.length == 2 && SERVICES.include?(parts[1]))

        occupied[path] = true
      end

      occupied
    end

    def paths_for(slug)
      [slug] + SERVICES.map { |svc| "#{slug}/#{svc}" }
    end
  end

  class TeachRetiredRedirectPage < Page
    def initialize(site, base, path)
      @site = site
      @base = base
      @dir = "teach/universities/#{path}"
      @name = "index.html"
      process(@name)
      self.data = {
        "layout" => "redirect",
        "redirect_to" => TeachRetiredRedirects::TARGET,
        "sitemap" => false,
        "robots" => "noindex, follow",
        "title" => "Redirecting…",
        "permalink" => "/teach/universities/#{path}/"
      }
      self.content = ""
    end
  end
end
