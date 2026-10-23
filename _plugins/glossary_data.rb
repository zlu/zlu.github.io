# frozen_string_literal: true

require "json"

# Emit /assets/data/{ai,med}-glossary.json for in-page hover tooltips.
module Jekyll
  class GlossaryData < Generator
    safe true
    priority :low

    COLLECTIONS = {
      "ai-glossary" => "ai-glossary.json",
      "med-glossary" => "med-glossary.json"
    }.freeze

    def generate(site)
      COLLECTIONS.each do |collection_name, filename|
        collection = site.collections[collection_name]
        next unless collection

        entries = {}
        collection.docs.each do |doc|
          slug = doc.data["slug"].to_s
          slug = File.basename(doc.path, ".*") if slug.empty?
          next if slug.empty?

          plain = strip_markdown(doc.content)
          entries[slug] = {
            "title" => doc.data["title"].to_s,
            "excerpt" => truncate(plain, 220),
            "url" => doc.url.to_s
          }
        end

        page = PageWithoutAFile.new(site, site.source, "assets/data", filename)
        page.content = JSON.pretty_generate(entries)
        page.data["layout"] = nil
        page.data["sitemap"] = false
        site.pages << page
      end
    end

    private

    def strip_markdown(text)
      text.to_s
          .gsub(/\{%.*?%\}/m, " ")
          .gsub(/\$\$(.*?)\$\$/m, ' \1 ')
          .gsub(/\$(.*?)\$/m, ' \1 ')
          .gsub(/```.*?```/m, " ")
          .gsub(/`[^`]*`/, " ")
          .gsub(/!\[[^\]]*\]\([^)]*\)/, " ")
          .gsub(/\[([^\]]*)\]\([^)]*\)/, '\1')
          .gsub(/[#>*_\-]+/, " ")
          .gsub(/\s+/, " ")
          .strip
    end

    def truncate(text, max)
      return text if text.length <= max

      cut = text[0, max]
      cut = cut.sub(/\s+\S*\z/, "")
      "#{cut}…"
    end
  end
end
