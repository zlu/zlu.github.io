# frozen_string_literal: true

# Enrich glossary collection docs for SEO / sitemap discoverability:
# description, keywords, sitemap flag, and a dedicated layout.
module Jekyll
  class GlossarySeo < Generator
    safe true
    priority :high

    COLLECTIONS = %w[ai-glossary med-glossary].freeze

    def generate(site)
      COLLECTIONS.each do |name|
        collection = site.collections[name]
        next unless collection

        collection.docs.each do |doc|
          enrich(doc, name)
        end
      end
    end

    private

    def enrich(doc, collection_name)
      doc.data["layout"] = "glossary" unless doc.data.key?("layout")
      doc.data["sitemap"] = true unless doc.data.key?("sitemap")

      # Avoid jekyll-seo-tag classifying collection docs as BlogPosting
      seo = doc.data["seo"].is_a?(Hash) ? doc.data["seo"] : {}
      seo["type"] ||= "WebPage"
      doc.data["seo"] = seo

      plain = strip_markdown(doc.content)
      doc.data["description"] ||= truncate(plain, 155)

      label = collection_name == "ai-glossary" ? "AI glossary" : "medicine glossary"
      parts = [doc.data["title"]]
      parts.concat(Array(doc.data["synonyms"]))
      parts.concat(Array(doc.data["term-en"])) if doc.data["term-en"]
      parts << label
      parts << "machine learning" if collection_name == "ai-glossary"
      parts << "LLM" if collection_name == "ai-glossary"
      doc.data["keywords"] ||= parts.compact.map(&:to_s).uniq.join(", ")
    end

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
