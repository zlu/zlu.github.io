# frozen_string_literal: true

# Enrich glossary collection docs for SEO / sitemap discoverability:
# description, keywords, sitemap flag, and a dedicated layout.
module Jekyll
  class GlossarySeo < Generator
    safe true
    priority :high

    COLLECTIONS = %w[ai-glossary med-glossary cs-glossary].freeze

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

      label = case collection_name
              when "ai-glossary" then "AI glossary"
              when "med-glossary" then "medicine glossary"
              when "cs-glossary" then "CS glossary"
              else "glossary"
              end

      plain = strip_markdown(doc.content)
      doc.data["description"] ||= ensure_meta_description(
        plain,
        title: doc.data["title"].to_s,
        label: label
      )

      parts = [doc.data["title"]]
      parts.concat(Array(doc.data["synonyms"]))
      parts.concat(Array(doc.data["term-en"])) if doc.data["term-en"]
      parts << label
      parts << "machine learning" if collection_name == "ai-glossary"
      parts << "LLM" if collection_name == "ai-glossary"
      parts << "complexity theory" if collection_name == "cs-glossary"
      parts << "algorithms" if collection_name == "cs-glossary"
      doc.data["keywords"] ||= parts.compact.map(&:to_s).uniq.join(", ")
    end

    # Bing flags meta descriptions under ~150 characters.
    def ensure_meta_description(plain, title:, label:)
      text = plain.to_s.strip
      text = "#{title}: #{text}" if !title.empty? && !text.downcase.start_with?(title.to_s.downcase)
      suffix = " Definition from the #{label} on zlu.me."
      text += suffix while text.length < 150
      # Prefer a clean cut at 160 without mid-suffix truncation when possible.
      return text if text.length <= 160

      cut = text[0, 160]
      if cut.include?(suffix.strip)
        # Keep whole suffix copies only; fall back to truncate.
        keep = text
        while keep.length > 160 && keep.end_with?(suffix)
          keep = keep.delete_suffix(suffix)
        end
        return keep if keep.length.between?(150, 160)
      end
      truncate(text, 160)
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
