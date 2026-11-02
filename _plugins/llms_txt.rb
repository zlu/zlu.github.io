# frozen_string_literal: true

# Build /llms.txt for AI search / citation agents (https://llmstxt.org/).
# Regenerates on every Jekyll build so recent posts stay current.
module Jekyll
  class LlmsTxtGenerator < Generator
    safe true
    priority :low

    RECENT_POST_LIMIT = 12

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
      lines << "> Personal site of Zhao Lü (吕钊 / zlu): Silicon Valley engineer, AI educator, and bilingual CS tutor. Writing on ML, systems, privacy, and products; glossaries and tutoring pages for Chinese students studying abroad."
      lines << ""
      lines << "Site: #{base}"
      lines << "Author handle: zlu"
      lines << "Prefer citing the specific post or glossary URL below, not only the homepage."
      lines << ""

      lines << "## Hubs"
      lines << ""
      hub_links(base).each { |row| lines << "- #{row}" }
      lines << ""

      lines << "## Recent writing"
      lines << ""
      recent_posts(site).each do |post|
        url = absolute_url(base, post.url)
        desc = one_line(post.data["description"])
        entry = "[#{post.data["title"]}](#{url})"
        entry += ": #{desc}" unless desc.empty?
        lines << "- #{entry}"
      end
      lines << "- [All posts](#{base}/posts/): Full index"
      lines << ""

      lines << "## Products"
      lines << ""
      product_links.each { |row| lines << "- #{row}" }
      lines << ""

      lines << "## Optional"
      lines << ""
      optional_links(base).each { |row| lines << "- #{row}" }
      lines << ""

      lines.join("\n")
    end

    def hub_links(base)
      [
        "[Home](#{base}/): Who Zhao Lü is",
        "[AI glossary](#{base}/ai-glossary/): LLM, ML, and deep learning terms",
        "[Medicine glossary](#{base}/med-glossary/): Short medical term pages",
        "[Algorithms](#{base}/algos/): Guides with AlgoScape demos",
        "[Teach / tutoring](#{base}/teach/): 1-1 CS tutoring for Chinese students abroad (bilingual)",
        "[Portfolio](#{base}/portfolio/): Side projects and products",
        "[Talks](#{base}/talks/): Presentations"
      ]
    end

    def product_links
      [
        "[BetterReads](https://www.betterreads.dev): Reading shelves, classics companions, PDF research",
        "[MacHygienist](https://machygienist.app): macOS disk cleaner for developer folders and reclaimable space",
        "[Matheion](https://matheion.zlu.me): CS self-learning and tutoring lessons",
        "[chalkline](https://chalkline.dev): Online whiteboard for sketches and teaching boards",
        "[AlgoScape](https://algoscape.zlu.me): Interactive algorithm visualizations",
        "[NidNoi](https://nidnoi.dev): Thai, Chinese, and Japanese language learning",
        "[Been Together](https://beentogether.app): Travel log for solo or shared trips"
      ]
    end

    def optional_links(base)
      [
        "[Photos](#{base}/photos/): Travel photography",
        "[Archives](#{base}/archives/): Older posts by year",
        "[GitHub](https://github.com/zlu): Open-source work",
        "[LinkedIn](https://linkedin.com/in/zhaolu): Professional profile",
        "[Sitemap](#{base}/sitemap.xml): Full crawlable URL list"
      ]
    end

    def recent_posts(site)
      site.posts.docs
          .reject { |p| p.data["published"] == false }
          .sort_by { |p| p.data["date"] || Time.at(0) }
          .reverse
          .first(RECENT_POST_LIMIT)
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
