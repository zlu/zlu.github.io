# frozen_string_literal: true

# Build /llms.txt for AI search / citation agents (https://llmstxt.org/).
# Regenerates on every Jekyll build. Runs after TeachPages (:high) so teach
# university/course data is available.
module Jekyll
  class LlmsTxtGenerator < Generator
    safe true
    priority :low

    MATHEION_LEARN = [
      ["Python", "https://matheion.zlu.me/learn/python/"],
      ["Java", "https://matheion.zlu.me/learn/java/"],
      ["SQL", "https://matheion.zlu.me/learn/sql/"],
      ["Web", "https://matheion.zlu.me/learn/web/"],
      ["CLI", "https://matheion.zlu.me/learn/cli/"]
    ].freeze

    def generate(site)
      page = PageWithoutAFile.new(site, site.source, "", "llms.txt")
      page.content = render(site)
      page.data["layout"] = nil
      page.data["sitemap"] = false
      site.pages << page
    end

    private

    def render(site)
      base = site.config["url"].to_s.sub(%r{/\z}, "")
      lines = []

      lines << "# Zhao Lü"
      lines << ""
      lines << "> Personal site of Zhao Lü (吕钊 / zlu): Silicon Valley engineer, AI educator, and bilingual CS tutor. Writing on ML, systems, privacy, and products; AI/medicine glossaries; algorithm guides; tutoring pages for Chinese students studying abroad; Matheion CS lessons."
      lines << ""
      lines << "Site: #{base}"
      lines << "Author handle: zlu"
      lines << "Cite the specific post, glossary term, teach course, or Matheion lesson URL when possible, not only the homepage."
      lines << "This file lists the citation-worthy pages. Thin tag indexes and retired teach redirect shells are omitted; see the sitemap for the full crawl set."
      lines << ""

      section(lines, "Hubs") { hub_links(base).each { |row| lines << "- #{row}" } }

      section(lines, "Writing") do
        all_posts(site).each do |post|
          lines << "- #{md_link(post.data["title"], absolute_url(base, post.url), post.data["description"])}"
        end
      end

      section(lines, "AI glossary") do
        lines << "- #{md_link("AI glossary index", "#{base}/ai-glossary/", "All LLM / ML / deep learning terms")}"
        collection_docs(site, "ai-glossary").each do |doc|
          lines << "- #{md_link(doc.data["title"], absolute_url(base, doc.url), doc.data["description"])}"
        end
      end

      section(lines, "Medicine glossary") do
        lines << "- #{md_link("Medicine glossary index", "#{base}/med-glossary/", "Short medical term pages")}"
        collection_docs(site, "med-glossary").each do |doc|
          lines << "- #{md_link(doc.data["title"], absolute_url(base, doc.url), doc.data["description"])}"
        end
      end

      section(lines, "Algorithms") do
        lines << "- #{md_link("Algorithms index", "#{base}/algos/", "Guides with AlgoScape demos")}"
        collection_docs(site, "algos").each do |doc|
          lines << "- #{md_link(doc.data["title"], absolute_url(base, doc.url), doc.data["description"])}"
        end
      end

      section(lines, "Teach / tutoring") do
        teach_links(site, base).each { |row| lines << "- #{row}" }
      end

      section(lines, "Teach universities") do
        teach_universities(site).each do |uni|
          name = uni["name"].to_s
          slug = uni["slug"].to_s
          next if name.empty? || slug.empty?

          note = one_line(uni["coverage_summary_en"])
          note = "#{uni["courses"].size} courses" if note.empty? && uni["courses"].is_a?(Array)
          lines << "- #{md_link(name, "#{base}/teach/universities/#{slug}/", note)}"
        end
      end

      section(lines, "Teach courses") do
        teach_universities(site).each do |uni|
          uni_name = uni["name"].to_s
          uni_slug = uni["slug"].to_s
          Array(uni["courses"]).each do |course|
            code = course["code"].to_s.strip
            title = course["title"].to_s.strip
            slug = course["slug"].to_s
            next if slug.empty?

            label = [code, title].reject(&:empty?).join(" — ")
            label = slug if label.empty?
            label = "#{uni_name}: #{label}" unless uni_name.empty?
            lines << "- #{md_link(label, "#{base}/teach/courses/#{uni_slug}/#{slug}/")}"
          end
        end
      end

      section(lines, "Matheion") do
        lines << "- #{md_link("Matheion", "https://matheion.zlu.me/", "CS self-learning and tutoring app / lessons")}"
        lines << "- #{md_link("Matheion learn hub", "https://matheion.zlu.me/learn/", "Open programming lesson tracks")}"
        MATHEION_LEARN.each do |name, url|
          lines << "- #{md_link("Matheion #{name}", url, "Open #{name} lessons")}"
        end
      end

      section(lines, "Products") { product_links.each { |row| lines << "- #{row}" } }

      section(lines, "Optional") { optional_links(base).each { |row| lines << "- #{row}" } }

      lines.join("\n")
    end

    def section(lines, title)
      lines << "## #{title}"
      lines << ""
      yield
      lines << ""
    end

    def hub_links(base)
      [
        md_link("Home", "#{base}/", "Who Zhao Lü is"),
        md_link("Posts index", "#{base}/posts/", "All blog posts"),
        md_link("AI glossary", "#{base}/ai-glossary/", "LLM, ML, and deep learning terms"),
        md_link("Medicine glossary", "#{base}/med-glossary/", "Short medical term pages"),
        md_link("Algorithms", "#{base}/algos/", "Guides with AlgoScape demos"),
        md_link("Teach / tutoring", "#{base}/teach/", "1-1 CS tutoring for Chinese students abroad (bilingual)"),
        md_link("Teach universities", "#{base}/teach/universities/", "University tutoring landing pages"),
        md_link("Portfolio", "#{base}/portfolio/", "Side projects and products"),
        md_link("Talks", "#{base}/talks/", "Presentations")
      ]
    end

    def teach_links(site, base)
      rows = [
        md_link("Teach hub", "#{base}/teach/", "Tutoring overview, WeChat booking, bilingual"),
        md_link("Teach FAQ", "#{base}/teach/faq/", "Common tutoring questions"),
        md_link("Universities index", "#{base}/teach/universities/", "All evidence-backed university pages")
      ]

      services = site.data["teach_services"]
      if services.is_a?(Hash)
        services.each do |key, meta|
          label = meta.is_a?(Hash) ? (meta["title_en"] || meta["en"] || key) : key
          desc = meta.is_a?(Hash) ? meta["desc_en"] : nil
          rows << md_link(label, "#{base}/teach/services/#{key}/", desc)
        end
      end

      teach_universities(site)
        .map { |u| u["region"].to_s }
        .reject(&:empty?)
        .uniq
        .sort
        .each do |region|
          rows << md_link("Teach region: #{region}", "#{base}/teach/#{region}/", "Universities in #{region}")
        end

      rows
    end

    def product_links
      [
        md_link("BetterReads", "https://www.betterreads.dev", "Reading shelves, classics companions, PDF research"),
        md_link("MacHygienist", "https://machygienist.app", "macOS disk cleaner for developer folders and reclaimable space"),
        md_link("chalkline", "https://chalkline.dev", "Online whiteboard for sketches and teaching boards"),
        md_link("AlgoScape", "https://algoscape.zlu.me", "Interactive algorithm visualizations"),
        md_link("NidNoi", "https://nidnoi.dev", "Thai, Chinese, and Japanese language learning"),
        md_link("Been Together", "https://beentogether.app", "Travel log for solo or shared trips"),
        md_link("HourHustle", "https://hh.zlu.me", "Freelancing time tracking"),
        md_link("Pixelator", "https://pixelator.zlu.me", "Mask sensitive information in PDFs"),
        md_link("LucidLens", "https://lucidlens.zlu.me", "Check bias in news sources"),
        md_link("AI Detector", "https://ai-detector.zlu.me", "Check AI authorship in text")
      ]
    end

    def optional_links(base)
      [
        md_link("Photos", "#{base}/photos/", "Travel photography"),
        md_link("Archives", "#{base}/archives/", "Older posts by year"),
        md_link("GitHub", "https://github.com/zlu", "Open-source work"),
        md_link("LinkedIn", "https://linkedin.com/in/zhaolu", "Professional profile"),
        md_link("Sitemap", "#{base}/sitemap.xml", "Full crawlable URL list")
      ]
    end

    def all_posts(site)
      site.posts.docs
          .reject { |p| p.data["published"] == false }
          .sort_by { |p| p.data["date"] || Time.at(0) }
          .reverse
    end

    def collection_docs(site, name)
      collection = site.collections[name]
      return [] unless collection

      collection.docs
                .reject { |d| d.data["published"] == false }
                .sort_by { |d| d.data["title"].to_s.downcase }
    end

    def teach_universities(site)
      Array(site.data["teach_universities"])
    end

    def md_link(title, url, note = nil)
      entry = "[#{one_line(title)}](#{url})"
      note = one_line(note)
      entry += ": #{note}" unless note.empty?
      entry
    end

    def absolute_url(base, path)
      return path if path.to_s.start_with?("http://", "https://")

      "#{base}#{path}"
    end

    def one_line(text)
      text.to_s.gsub(/\s+/, " ").strip
    end
  end
end
