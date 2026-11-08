# frozen_string_literal: true

# Give each /tags/:name/ page a unique meta description.
# Layout front matter is not merged into page.data for jekyll-seo-tag, and
# _tags/*.md stubs only set tag/permalink — without this they all inherit
# site.description and trip Bing/Google "duplicate meta description" reports.
module Jekyll
  class TagSeo < Generator
    safe true
    priority :high

    def generate(site)
      collection = site.collections["tags"]
      return unless collection

      collection.docs.each { |doc| enrich(doc) }
    end

    private

    def enrich(doc)
      tag = doc.data["tag"].to_s.strip
      return if tag.empty?

      doc.data["description"] ||=
        "Articles and tutorials tagged \"#{tag}\" by Zhao Lü — AI, machine learning, " \
        "data science, and systems programming on zlu.me."

      seo = doc.data["seo"].is_a?(Hash) ? doc.data["seo"] : {}
      seo["type"] ||= "WebPage"
      doc.data["seo"] = seo
    end
  end
end
