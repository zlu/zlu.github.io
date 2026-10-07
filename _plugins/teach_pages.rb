# frozen_string_literal: true

module Jekyll
  # Generates SEO landing pages for tutoring universities, courses, and regions.
  class TeachPagesGenerator < Generator
    safe true
    priority :high

    REGION_MAP = {
      "australia" => {
        "en" => "Australia",
        "cn" => "澳大利亚",
        "match" => /
          melbourne|monash|sydney|unsw|macquarie|queensland|western.?australia|
          eynesbury|south.?australia
        /xi
      },
      "uk" => {
        "en" => "United Kingdom",
        "cn" => "英国",
        "match" => /
          manchester|imperial|nottingham|birmingham|glasgow|liverpool|southampton|
          stirling|warwick|leeds|cardiff|queen.?mary|brunel|bayes|edingburgh|edinburgh
        /xi
      },
      "usa" => {
        "en" => "United States",
        "cn" => "美国",
        "match" => /
          carnegie|lehigh|northeastern|ohio|pittsburgh|rochester|san.?jose|
          uc.?irvine|uiuc|william.?mary|john.?hopkins
        /xi
      },
      "canada" => {
        "en" => "Canada",
        "cn" => "加拿大",
        "match" => /concordia/i
      },
      "hong-kong" => {
        "en" => "Hong Kong",
        "cn" => "香港",
        "match" => /hong.?kong|hku|baptist|metropolitan|macau|north.?china/i
      },
      "singapore" => {
        "en" => "Singapore",
        "cn" => "新加坡",
        "match" => /singapore|ntu|nus|jcu|national.?university.?of.?singapore/i
      },
      "malaysia" => {
        "en" => "Malaysia",
        "cn" => "马来西亚",
        "match" => /malaysia/i
      },
      "new-zealand" => {
        "en" => "New Zealand",
        "cn" => "新西兰",
        "match" => /auckland/i
      }
    }.freeze

    def generate(site)
      courses_data = site.data["courses"]
      return unless courses_data.is_a?(Hash)

      universities = []
      slug_lookup = {}
      courses_data.each do |slug, entry|
        next unless entry.is_a?(Hash) && entry["university"]

        data_key = slug.to_s
        uni_slug = ascii_slug(data_key)
        next if uni_slug.empty?

        university = entry["university"].to_s.strip
        course_list = Array(entry["courses"]).map { |c| normalize_course(c) }.compact
        region_key = detect_region("#{data_key} #{uni_slug}", university)

        uni_page = build_university_page(site, uni_slug, university, course_list, region_key)
        site.pages << uni_page

        course_list.each do |course|
          site.pages << build_course_page(site, uni_slug, university, course, region_key)
        end

        record = {
          "slug" => uni_slug,
          "name" => university,
          "region" => region_key,
          "courses" => course_list
        }
        universities << record
        slug_lookup[data_key] = uni_slug
      end

      site.data["teach_universities"] = universities.sort_by { |u| u["name"].downcase }
      site.data["teach_uni_slugs"] = slug_lookup

      REGION_MAP.each_key do |region_key|
        region_unis = universities.select { |u| u["region"] == region_key }
        next if region_unis.empty?

        site.pages << build_region_page(site, region_key, region_unis)
      end

      site.pages << build_universities_index(site, universities)
      Jekyll.logger.info "TeachPages:", "generated #{universities.size} universities, " \
        "#{universities.sum { |u| u['courses'].size }} courses, " \
        "#{REGION_MAP.count { |k, _| universities.any? { |u| u['region'] == k } }} regions"
    end

    private

    def normalize_course(course)
      return nil unless course.is_a?(Hash)

      code = course["code"].to_s.strip
      title = course["title"].to_s.strip
      return nil if code.empty? && title.empty?

      {
        "code" => code,
        "title" => title,
        "department" => course["department"].to_s.strip,
        "slug" => course_slug(code, title)
      }
    end

    def course_slug(code, title)
      base = code.empty? ? title : code
      slug = ascii_slug(base)
      return slug unless slug.empty?

      require "digest"
      "course-#{Digest::MD5.hexdigest(base)[0, 8]}"
    end

    def ascii_slug(text)
      text.to_s.downcase
          .gsub(/[^a-z0-9]+/, "-")
          .gsub(/-+/, "-")
          .gsub(/^-|-$/, "")
    end

    def detect_region(slug, university)
      haystack = "#{slug} #{university}"
      REGION_MAP.each do |key, meta|
        return key if haystack.match?(meta["match"])
      end
      "international"
    end

    def region_label(key, lang)
      return (lang == "cn" ? "海外" : "International") if key == "international"

      REGION_MAP.fetch(key, {})[lang] || key
    end

    def build_university_page(site, uni_slug, university, courses, region_key)
      codes = courses.map { |c| c["code"] }.reject(&:empty?)
      title_en = "#{university} Computer Science Tutoring | AI & Programming"
      title_cn = "#{university} 计算机辅导 | 人工智能与编程一对一"
      description = "Bilingual CS tutoring for #{university} students. " \
                    "Coverage includes #{codes.first(6).join(', ')}#{codes.size > 6 ? ' and more' : ''}. " \
                    "WeChat booking · 中英双语."

      page = TeachGeneratedPage.new(site, site.source, "teach/universities/#{uni_slug}")
      page.data.merge!(
        "layout" => "teach_university",
        "title" => title_cn,
        "title_en" => title_en,
        "title_cn" => title_cn,
        "title_suffix" => "zlu.me/teach",
        "description" => description,
        "keywords" => "#{university}, CS tutoring, 留学生辅导, #{codes.first(8).join(', ')}, 计算机辅导",
        "university" => university,
        "uni_slug" => uni_slug,
        "courses" => courses,
        "region" => region_key,
        "region_en" => region_label(region_key, "en"),
        "region_cn" => region_label(region_key, "cn"),
        "lang" => "zh-CN",
        "sitemap" => true
      )
      page
    end

    def build_course_page(site, uni_slug, university, course, region_key)
      code = course["code"]
      title = course["title"]
      label = [code, title].reject(&:empty?).join(": ")
      title_en = "#{label} Tutoring | #{university} Computer Science Tutoring"
      title_cn = "#{label} 辅导 | #{university} 计算机一对一"
      description = "1-1 tutoring for #{label} at #{university}. " \
                    "Bilingual Chinese/English help with assignments, projects, and exam prep. No ghostwriting."

      page = TeachGeneratedPage.new(
        site,
        site.source,
        "teach/courses/#{uni_slug}/#{course['slug']}"
      )
      page.data.merge!(
        "layout" => "teach_course",
        "title" => title_cn,
        "title_en" => title_en,
        "title_cn" => title_cn,
        "title_suffix" => "zlu.me/teach",
        "description" => description,
        "keywords" => "#{code}, #{title}, #{university}, tutoring, 辅导, 留学生, AI, Python",
        "university" => university,
        "uni_slug" => uni_slug,
        "course_code" => code,
        "course_title" => title,
        "course_department" => course["department"],
        "course_slug" => course["slug"],
        "region" => region_key,
        "region_en" => region_label(region_key, "en"),
        "region_cn" => region_label(region_key, "cn"),
        "lang" => "zh-CN",
        "sitemap" => true
      )
      page
    end

    def build_region_page(site, region_key, universities)
      en = region_label(region_key, "en")
      cn = region_label(region_key, "cn")
      title_en = "CS Tutoring for Chinese Students in #{en} | Computer Science Tutoring"
      title_cn = "#{cn}留学生计算机辅导 | 中英双语一对一"
      course_count = universities.sum { |u| u["courses"].size }
      description = "Bilingual computer science tutoring for Chinese students in #{en}. " \
                    "#{universities.size} universities · #{course_count}+ courses. WeChat booking."

      page = TeachGeneratedPage.new(site, site.source, "teach/#{region_key}")
      page.data.merge!(
        "layout" => "teach_region",
        "title" => title_cn,
        "title_en" => title_en,
        "title_cn" => title_cn,
        "title_suffix" => "zlu.me/teach",
        "description" => description,
        "keywords" => "#{en}, #{cn}, 留学生辅导, computer science tutoring, AI, Python",
        "region" => region_key,
        "region_en" => en,
        "region_cn" => cn,
        "universities" => universities.sort_by { |u| u["name"].downcase },
        "lang" => "zh-CN",
        "sitemap" => true
      )
      page
    end

    def build_universities_index(site, universities)
      title_en = "University CS Tutoring Directory | Computer Science Tutoring"
      title_cn = "大学计算机辅导目录 | 留学生一对一"
      description = "Browse #{universities.size} universities for bilingual CS tutoring — " \
                    "AI, data science, and programming course support for Chinese students abroad."

      page = TeachGeneratedPage.new(site, site.source, "teach/universities")
      page.data.merge!(
        "layout" => "teach_universities_index",
        "title" => title_cn,
        "title_en" => title_en,
        "title_cn" => title_cn,
        "title_suffix" => "zlu.me/teach",
        "description" => description,
        "keywords" => "university tutoring directory, 留学生辅导, computer science, AI",
        "universities" => universities.sort_by { |u| u["name"].downcase },
        "lang" => "zh-CN",
        "sitemap" => true
      )
      page
    end
  end

  class TeachGeneratedPage < Page
    def initialize(site, base, dir)
      @site = site
      @base = base
      @dir = dir
      @name = "index.html"
      process(@name)
      self.data = {}
      self.content = ""
    end
  end
end
