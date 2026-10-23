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
          eynesbury|south.?australia|anu|australian.?national|uts|technology.?sydney|
          adelaide|rmit|wollongong
        /xi
      },
      "uk" => {
        "en" => "United Kingdom",
        "cn" => "英国",
        "match" => /
          manchester|imperial|nottingham|birmingham|glasgow|liverpool|southampton|
          stirling|warwick|leeds|cardiff|queen.?mary|brunel|bayes|edingburgh|edinburgh|
          ucl|university.?college.?london|kings.?college|bristol
        /xi
      },
      "usa" => {
        "en" => "United States",
        "cn" => "美国",
        "match" => /
          carnegie|lehigh|northeastern|ohio|pittsburgh|rochester|san.?jose|
          uc.?irvine|uiuc|william.?mary|john.?hopkins|berkeley|stanford|georgia.?tech|
          nyu|new.?york.?university|ucla|washington|umich|michigan|ut.?austin|texas.?at.?austin
        /xi
      },
      "canada" => {
        "en" => "Canada",
        "cn" => "加拿大",
        "match" => /concordia|toronto|ubc|british.?columbia|waterloo|mcgill|alberta/i
      },
      "hong-kong" => {
        "en" => "Hong Kong",
        "cn" => "香港",
        "match" => /hong.?kong|hku|baptist|metropolitan|macau|cuhk|chinese.?university|polyu|polytechnic/i
      },
      "singapore" => {
        "en" => "Singapore",
        "cn" => "新加坡",
        "match" => /singapore|ntu|nus|jcu|national.?university.?of.?singapore|smu|management.?university/i
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

    # Service verticals for tutoring SEO pages
    SERVICES = {
      "sync" => {
        "en" => "Course sync tutoring",
        "cn" => "同步课程辅导",
        "title_en" => "Course Sync Tutoring",
        "title_cn" => "同步课程辅导",
        "desc_en" => "Keep up with lectures in Chinese and English — syllabus walkthrough, weekly Q&A. Direct tutor, no middleman fee.",
        "desc_cn" => "中英双语跟课：大纲梳理、每周答疑，解决听不懂与掉队。直接约老师，无中介抽成。"
      },
      "assignment" => {
        "en" => "Assignment tutoring",
        "cn" => "作业辅导",
        "title_en" => "Assignment & Project Help",
        "title_cn" => "作业与 Project 辅导",
        "desc_en" => "Debug and approach review. You submit your own work — no ghostwriting, no middleman fee.",
        "desc_cn" => "拆题、debug、复盘思路。作业本人提交；不代写；无中介抽成。"
      },
      "exam" => {
        "en" => "Exam prep tutoring",
        "cn" => "考前突击辅导",
        "title_en" => "Exam Prep Tutoring",
        "title_cn" => "考前突击辅导",
        "desc_en" => "Topic maps, past-paper style drills, weak-spot sprints before midterms and finals.",
        "desc_cn" => "考点地图、类 past paper 练习与薄弱点突击，期中期末冲刺。"
      },
      "preview" => {
        "en" => "Course preview tutoring",
        "cn" => "课程预习辅导",
        "title_en" => "Course Preview Tutoring",
        "title_cn" => "课程预习辅导",
        "desc_en" => "Pre-term head start: programming basics, jargon, and assessment format before week 1.",
        "desc_cn" => "开学前抢跑：编程基础、专业术语与考核方式预热，降低第一周冲击。"
      }
    }.freeze

    # More specific themes first; "general" is the fallback (not listed here).
    THEME_MATCH_ORDER = %w[
      machine-learning
      data-science
      security
      databases
      math-for-cs
      algorithms
      systems
      web-mobile
      programming
    ].freeze

    def generate(site)
      courses_data = site.data["courses"]
      return unless courses_data.is_a?(Hash)

      themes = site.data["teach_themes"]
      themes = {} unless themes.is_a?(Hash)

      # Bulk directory from _data/teach_universities.yml (array of name/region/slug)
      bulk = site.data["teach_universities"]
      bulk = [] unless bulk.is_a?(Array)

      universities = []
      slug_lookup = {}
      seen_names = {}
      seen_slugs = {}
      rich_uni_payloads = []
      theme_index = Hash.new { |h, k| h[k] = [] }

      # 1) Collect rich universities + index courses by theme (pages built after index)
      courses_data.each do |slug, entry|
        next unless entry.is_a?(Hash) && entry["university"]

        data_key = slug.to_s
        uni_slug = ascii_slug(data_key)
        next if uni_slug.empty?

        university = entry["university"].to_s.strip
        name_key = university.downcase
        course_list = Array(entry["courses"]).map { |c| normalize_course(c, themes) }.compact
        region_key = detect_region("#{data_key} #{uni_slug} #{university}", university)

        next if seen_slugs[uni_slug]

        rich_uni_payloads << {
          "uni_slug" => uni_slug,
          "university" => university,
          "course_list" => course_list,
          "region_key" => region_key
        }

        course_list.each do |course|
          theme_index[course["theme"]] << {
            "uni_slug" => uni_slug,
            "university" => university,
            "course" => course
          }
        end

        record = {
          "slug" => uni_slug,
          "name" => university,
          "region" => region_key,
          "courses" => course_list,
          "rich" => true
        }
        universities << record
        slug_lookup[data_key] = uni_slug
        seen_slugs[uni_slug] = true
        seen_names[name_key] = uni_slug
      end

      rich_count = universities.size
      course_count = universities.sum { |u| u["courses"].size }
      site.data["teach_course_count"] = course_count

      reading_by_theme = {}
      themes.each do |theme_id, meta|
        next unless meta.is_a?(Hash)

        reading_by_theme[theme_id.to_s] = resolve_theme_reading(site, meta["reading"])
      end

      # Emit rich uni + service + course pages (courses need theme_index)
      rich_uni_payloads.each do |payload|
        uni_slug = payload["uni_slug"]
        university = payload["university"]
        course_list = payload["course_list"]
        region_key = payload["region_key"]

        site.pages << build_university_page(site, uni_slug, university, course_list, region_key, true)

        SERVICES.each_key do |service_key|
          site.pages << build_uni_service_page(
            site, uni_slug, university, course_list, region_key, service_key, true
          )
        end

        course_list.each do |course|
          site.pages << build_course_page(
            site,
            uni_slug,
            university,
            course,
            region_key,
            themes: themes,
            theme_index: theme_index,
            reading_by_theme: reading_by_theme,
            course_count: course_count,
            siblings: course_list
          )
        end
      end

      # 2) Bulk directory universities (+ service landings so uni→service links never 404)
      bulk_added = 0
      bulk.each do |entry|
        next unless entry.is_a?(Hash)

        university = entry["name"].to_s.strip
        next if university.empty?

        name_key = university.downcase
        next if seen_names[name_key]

        uni_slug = ascii_slug(entry["slug"].to_s)
        uni_slug = ascii_slug(university) if uni_slug.empty?
        next if uni_slug.empty? || seen_slugs[uni_slug]

        region_key = entry["region"].to_s
        region_key = detect_region("#{uni_slug} #{university}", university) if region_key.empty?
        region_key = "international" if region_key.empty?

        site.pages << build_university_page(site, uni_slug, university, [], region_key, true)

        # Service landings for directory unis too — university pages link these.
        SERVICES.each_key do |service_key|
          site.pages << build_uni_service_page(
            site, uni_slug, university, [], region_key, service_key, true
          )
        end

        universities << {
          "slug" => uni_slug,
          "name" => university,
          "region" => region_key,
          "courses" => [],
          "rich" => false
        }
        seen_slugs[uni_slug] = true
        seen_names[name_key] = uni_slug
        slug_lookup[uni_slug] = uni_slug
        bulk_added += 1
      end

      universities.sort_by! { |u| u["name"].downcase }
      site.data["teach_universities"] = universities
      site.data["teach_uni_slugs"] = slug_lookup
      site.data["teach_services"] = SERVICES
      site.data["teach_uni_count"] = universities.size
      site.data["teach_course_count"] = course_count

      REGION_MAP.each_key do |region_key|
        region_unis = universities.select { |u| u["region"] == region_key }
        next if region_unis.empty?

        site.pages << build_region_page(site, region_key, region_unis)
      end

      # Service hubs list course-backed schools (full directory is on /teach/universities/)
      hub_unis = universities.select do |u|
        u["courses"].size.positive? &&
          %w[australia uk usa canada hong-kong singapore].include?(u["region"])
      end
      SERVICES.each_key do |service_key|
        site.pages << build_service_hub_page(site, service_key, hub_unis)
      end

      site.pages << build_universities_index(site, universities)
      uni_service_count = universities.size * SERVICES.size
      theme_counts = theme_index.transform_values(&:size)
      Jekyll.logger.info "TeachPages:", "generated #{universities.size} universities " \
        "(#{rich_count} with courses, #{bulk_added} directory), " \
        "#{course_count} course pages, #{uni_service_count} uni-service pages, " \
        "#{SERVICES.size} service hubs, " \
        "#{REGION_MAP.count { |k, _| universities.any? { |u| u['region'] == k } }} regions; " \
        "themes=#{theme_counts.map { |k, v| "#{k}:#{v}" }.join(', ')}"
    end

    private

    def normalize_course(course, themes = {})
      return nil unless course.is_a?(Hash)

      code = course["code"].to_s.strip
      title = course["title"].to_s.strip
      return nil if code.empty? && title.empty?

      override = course["theme"].to_s.strip
      theme_id = if !override.empty? && themes.key?(override)
                   override
                 else
                   infer_theme("#{code} #{title}", themes)
                 end

      {
        "code" => code,
        "title" => title,
        "department" => course["department"].to_s.strip,
        "slug" => course_slug(code, title),
        "theme" => theme_id
      }
    end

    def infer_theme(haystack, themes)
      text = haystack.to_s.downcase
      THEME_MATCH_ORDER.each do |theme_id|
        meta = themes[theme_id]
        next unless meta.is_a?(Hash)

        pattern = meta["match"].to_s.gsub(/\s+/, "")
        next if pattern.empty?

        return theme_id if text.match?(Regexp.new(pattern, Regexp::IGNORECASE))
      end
      "general"
    end

    def resolve_theme_reading(site, reading_list)
      Array(reading_list).filter_map do |item|
        next unless item.is_a?(Hash)

        kind = item["kind"].to_s
        name = item["name"].to_s.strip
        next if name.empty?

        url = case kind
              when "glossary"
                find_glossary_url(site, name)
              when "post"
                find_post_url(site, name)
              end
        next unless url

        {
          "url" => url,
          "title_en" => item["title_en"].to_s,
          "title_cn" => item["title_cn"].to_s
        }
      end
    end

    def find_glossary_url(site, name)
      docs = site.collections["ai-glossary"]&.docs
      return nil unless docs

      doc = docs.find do |d|
        File.basename(d.path, ".*") == name || d.data["slug"].to_s == name
      end
      doc&.url
    end

    def find_post_url(site, name)
      posts = site.posts.docs.select do |p|
        base = File.basename(p.path, ".*").sub(/^\d{4}-\d{2}-\d{2}-/, "")
        base == name ||
          base == "#{name}-zh" ||
          base == "#{name}-cn" ||
          base.end_with?(name) ||
          base.include?(name)
      end
      return nil if posts.empty?

      preferred = posts.find do |p|
        base = File.basename(p.path, ".*").sub(/^\d{4}-\d{2}-\d{2}-/, "")
        base == name
      end
      preferred ||= posts.find do |p|
        !File.basename(p.path, ".*").match?(/-(zh|cn)$/)
      end
      (preferred || posts.first)&.url
    end

    def related_course_payload(rec)
      course = rec["course"]
      {
        "uni_slug" => rec["uni_slug"],
        "university" => rec["university"],
        "code" => course["code"],
        "title" => course["title"],
        "slug" => course["slug"],
        "url" => "/teach/courses/#{rec['uni_slug']}/#{course['slug']}/"
      }
    end

    def related_same_uni(siblings, course, limit = 4)
      others = siblings.reject { |c| c["slug"] == course["slug"] }
      same_theme = others.select { |c| c["theme"] == course["theme"] }
      rest = others - same_theme
      (same_theme + rest).first(limit).map do |c|
        {
          "uni_slug" => nil,
          "university" => nil,
          "code" => c["code"],
          "title" => c["title"],
          "slug" => c["slug"],
          "url" => nil,
          "theme" => c["theme"]
        }
      end
    end

    def related_same_theme(theme_index, theme_id, uni_slug, limit = 6)
      pool = theme_index.fetch(theme_id, []).reject { |r| r["uni_slug"] == uni_slug }
      by_uni = pool.group_by { |r| r["uni_slug"] }
      ordered_unis = by_uni.keys.sort
      picked = []
      idx = 0
      while picked.size < limit && ordered_unis.any?
        u = ordered_unis[idx % ordered_unis.size]
        rec = by_uni[u].shift
        picked << related_course_payload(rec) if rec
        ordered_unis.delete(u) if by_uni[u].empty?
        idx += 1
        break if idx > 10_000
      end
      picked
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

    def build_university_page(site, uni_slug, university, courses, region_key, in_sitemap = true)
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
        "sitemap" => in_sitemap
      )
      page
    end

    def build_course_page(
      site, uni_slug, university, course, region_key,
      themes:, theme_index:, reading_by_theme:, course_count:, siblings:
    )
      code = course["code"]
      title = course["title"]
      label = [code, title].reject(&:empty?).join(": ")
      theme_id = course["theme"].to_s
      theme_id = "general" if theme_id.empty?
      theme_meta = themes[theme_id]
      theme_meta = themes["general"] || {} unless theme_meta.is_a?(Hash)

      theme_label_en = theme_meta["label_en"].to_s
      theme_label_cn = theme_meta["label_cn"].to_s
      topics = Array(theme_meta["topics"]).select { |t| t.is_a?(Hash) }
      topic_bits = topics.first(4).map { |t| t["en"].to_s }.reject(&:empty?)

      title_en = "#{label} Tutoring | #{university} Computer Science Tutoring"
      # Match competitor title pattern: 大学 + 课号 + 辅导
      title_cn = if code.empty?
                   "#{university} #{title} 课程辅导 | 中英双语一对一"
                 else
                   "#{university} #{code} 课程辅导 | #{title}"
                 end
      description = "1-1 tutoring for #{label} at #{university}"
      description += " (#{theme_label_en})" unless theme_label_en.empty?
      description += ". "
      description += "Typical focus: #{topic_bits.join('; ')}. " unless topic_bits.empty?
      description += "Bilingual Chinese/English — assignments, projects, exam prep. No ghostwriting."

      topic_kw = topics.first(6).flat_map { |t| [t["en"], t["cn"]] }.compact.map(&:to_s).reject(&:empty?)
      keywords = [
        code, title, university, theme_label_en, theme_label_cn,
        "作业辅导", "考前辅导", "同步辅导", "留学生", "AI", "Python",
        *topic_kw.first(8)
      ].reject(&:empty?).join(", ")

      same_uni = related_same_uni(siblings, course, 4).map do |c|
        c.merge(
          "uni_slug" => uni_slug,
          "university" => university,
          "url" => "/teach/courses/#{uni_slug}/#{c['slug']}/"
        )
      end
      same_theme = related_same_theme(theme_index, theme_id, uni_slug, 6)
      theme_reading = Array(reading_by_theme[theme_id]).first(4)

      about_topics = topics.first(8).map { |t| t["en"].to_s }.reject(&:empty?)

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
        "keywords" => keywords,
        "university" => university,
        "uni_slug" => uni_slug,
        "course_code" => code,
        "course_title" => title,
        "course_department" => course["department"],
        "course_slug" => course["slug"],
        "theme" => theme_id,
        "theme_label_en" => theme_label_en,
        "theme_label_cn" => theme_label_cn,
        "theme_lead_en" => theme_meta["lead_en"].to_s,
        "theme_lead_cn" => theme_meta["lead_cn"].to_s,
        "theme_topics" => topics,
        "theme_stuck" => Array(theme_meta["stuck"]).select { |t| t.is_a?(Hash) },
        "theme_session_angles" => Array(theme_meta["session_angles"]).select { |t| t.is_a?(Hash) },
        "theme_reading" => theme_reading,
        "theme_about" => about_topics,
        "related_same_uni" => same_uni,
        "related_same_theme" => same_theme,
        "teach_course_count" => course_count,
        "region" => region_key,
        "region_en" => region_label(region_key, "en"),
        "region_cn" => region_label(region_key, "cn"),
        "lang" => "zh-CN",
        "sitemap" => true
      )
      page
    end

    def build_uni_service_page(site, uni_slug, university, courses, region_key, service_key, in_sitemap = true)
      svc = SERVICES.fetch(service_key)
      title_en = "#{university} #{svc['title_en']} | Computer Science Tutoring"
      title_cn = "#{university}#{svc['title_cn']} | 计算机留学生一对一"
      codes = courses.map { |c| c["code"] }.reject(&:empty?)
      description = "#{svc['desc_en']} Popular at #{university}: #{codes.first(5).join(', ')}."

      page = TeachGeneratedPage.new(
        site,
        site.source,
        "teach/universities/#{uni_slug}/#{service_key}"
      )
      page.data.merge!(
        "layout" => "teach_uni_service",
        "title" => title_cn,
        "title_en" => title_en,
        "title_cn" => title_cn,
        "title_suffix" => "zlu.me/teach",
        "description" => description,
        "keywords" => "#{university}, #{svc['cn']}, #{svc['en']}, 留学生辅导, #{codes.first(6).join(', ')}",
        "university" => university,
        "uni_slug" => uni_slug,
        "courses" => courses,
        "service_key" => service_key,
        "service_en" => svc["en"],
        "service_cn" => svc["cn"],
        "service_title_en" => svc["title_en"],
        "service_title_cn" => svc["title_cn"],
        "service_desc_en" => svc["desc_en"],
        "service_desc_cn" => svc["desc_cn"],
        "region" => region_key,
        "region_en" => region_label(region_key, "en"),
        "region_cn" => region_label(region_key, "cn"),
        "lang" => "zh-CN",
        "sitemap" => in_sitemap
      )
      page
    end

    def build_service_hub_page(site, service_key, universities)
      svc = SERVICES.fetch(service_key)
      title_en = "#{svc['title_en']} for Chinese Students Abroad | CS Tutoring"
      title_cn = "留学生#{svc['title_cn']} | 计算机中英双语一对一"
      # Prefer schools with course pages at the top of the hub list
      ordered = universities.sort_by do |u|
        [(u["courses"].size.positive? ? 0 : 1), u["name"].downcase]
      end
      with_courses = ordered.count { |u| u["courses"].size.positive? }
      description = "#{svc['desc_en']} #{with_courses}+ course-mapped universities; " \
                    "#{ordered.size} schools across AU/UK/US/CA/HK/SG."

      page = TeachGeneratedPage.new(site, site.source, "teach/services/#{service_key}")
      page.data.merge!(
        "layout" => "teach_service_hub",
        "title" => title_cn,
        "title_en" => title_en,
        "title_cn" => title_cn,
        "title_suffix" => "zlu.me/teach",
        "description" => description,
        "keywords" => "#{svc['cn']}, #{svc['en']}, 留学生辅导, 计算机, AI, Python",
        "service_key" => service_key,
        "service_en" => svc["en"],
        "service_cn" => svc["cn"],
        "service_title_en" => svc["title_en"],
        "service_title_cn" => svc["title_cn"],
        "service_desc_en" => svc["desc_en"],
        "service_desc_cn" => svc["desc_cn"],
        "universities" => ordered,
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
