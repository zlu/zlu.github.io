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

    def generate(site)
      courses_data = site.data["courses"]
      return unless courses_data.is_a?(Hash)

      # Bulk directory from _data/teach_universities.yml (array of name/region/slug)
      bulk = site.data["teach_universities"]
      bulk = [] unless bulk.is_a?(Array)

      universities = []
      slug_lookup = {}
      seen_names = {}
      seen_slugs = {}

      # 1) Rich universities from course YAML (pages + services + course landings)
      courses_data.each do |slug, entry|
        next unless entry.is_a?(Hash) && entry["university"]

        data_key = slug.to_s
        uni_slug = ascii_slug(data_key)
        next if uni_slug.empty?

        university = entry["university"].to_s.strip
        name_key = university.downcase
        course_list = Array(entry["courses"]).map { |c| normalize_course(c) }.compact
        region_key = detect_region("#{data_key} #{uni_slug} #{university}", university)

        next if seen_slugs[uni_slug]

        uni_page = build_university_page(site, uni_slug, university, course_list, region_key)
        site.pages << uni_page

        SERVICES.each_key do |service_key|
          site.pages << build_uni_service_page(
            site, uni_slug, university, course_list, region_key, service_key
          )
        end

        course_list.each do |course|
          site.pages << build_course_page(site, uni_slug, university, course, region_key)
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

      # 2) Bulk directory universities (landing page only — keeps sitemap/build scalable)
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

        site.pages << build_university_page(site, uni_slug, university, [], region_key)

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

      REGION_MAP.each_key do |region_key|
        region_unis = universities.select { |u| u["region"] == region_key }
        next if region_unis.empty?

        site.pages << build_region_page(site, region_key, region_unis)
      end

      # Service hubs highlight universities that already have course-level pages
      rich_unis = universities.select { |u| u["rich"] }
      SERVICES.each_key do |service_key|
        site.pages << build_service_hub_page(site, service_key, rich_unis)
      end

      site.pages << build_universities_index(site, universities)
      Jekyll.logger.info "TeachPages:", "generated #{universities.size} universities " \
        "(#{rich_count} with courses, #{bulk_added} directory), " \
        "#{course_count} course pages, #{rich_count * SERVICES.size} uni-service pages, " \
        "#{SERVICES.size} service hubs, " \
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
      # Match competitor title pattern: 大学 + 课号 + 辅导
      title_cn = if code.empty?
                   "#{university} #{title} 课程辅导 | 中英双语一对一"
                 else
                   "#{university} #{code} 课程辅导 | #{title}"
                 end
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
        "keywords" => "#{code}, #{title}, #{university}, 作业辅导, 考前辅导, 同步辅导, 留学生, AI, Python",
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

    def build_uni_service_page(site, uni_slug, university, courses, region_key, service_key)
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
        "sitemap" => true
      )
      page
    end

    def build_service_hub_page(site, service_key, universities)
      svc = SERVICES.fetch(service_key)
      title_en = "#{svc['title_en']} for Chinese Students Abroad | CS Tutoring"
      title_cn = "留学生#{svc['title_cn']} | 计算机中英双语一对一"
      focus = universities.select do |u|
        %w[australia uk usa canada hong-kong singapore].include?(u["region"])
      end
      description = "#{svc['desc_en']} Covering #{focus.size} universities across AU/UK/US/CA/HK/SG."

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
        "universities" => focus.sort_by { |u| u["name"].downcase },
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
