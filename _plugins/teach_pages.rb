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
          \bucl\b|university.?college.?london|kings.?college|bristol|durham|newcastle
        /xi
      },
      "usa" => {
        "en" => "United States",
        "cn" => "美国",
        "match" => /
          carnegie|lehigh|northeastern|ohio|pittsburgh|rochester|san.?jose|
          uc.?irvine|uiuc|william.?mary|john.?hopkins|berkeley|stanford|georgia.?tech|
          nyu|new.?york.?university|ucla|\buc\b.?los.?angeles|washington|umich|michigan|
          ut.?austin|texas.?at.?austin|cornell|baruch|oregon.?state|unc|chapel.?hill|
          rensselaer|rpi|\bucsb\b|santa.?barbara|university.?of.?california|
          illinois.?urbana|champaign|\bneu\b|northeastern.?university
        /xi
      },
      "canada" => {
        "en" => "Canada",
        "cn" => "加拿大",
        "match" => /
          concordia|toronto|ubc|british.?columbia|waterloo|mcgill|alberta|
          algonquin|western.?university
        /xi
      },
      "china" => {
        "en" => "China / Macau",
        "cn" => "中国 / 澳门",
        "match" => /
          beijing.?normal|north.?china|nanyang.?institute|macau|macao|\bmust\b|
          city.?university.?of.?macau|duke.?kunshan|cuhk.?shenzhen
        /xi
      },
      "hong-kong" => {
        "en" => "Hong Kong",
        "cn" => "香港",
        "match" => /
          hong.?kong|hku|hkust|baptist|metropolitan|cuhk|chinese.?university|
          \bpolyu\b|hk.?poly
        /xi
      },
      "singapore" => {
        "en" => "Singapore",
        "cn" => "新加坡",
        "match" => /
          singapore|ntu|\bnus\b|jcu|national.?university.?of.?singapore|smu|
          management.?university|psb.?academy
        /xi
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
      },
      "germany" => {
        "en" => "Germany",
        "cn" => "德国",
        "match" => /munich|tum|lmu|technical.?university.?of.?munich|technische/i
      },
      "ireland" => {
        "en" => "Ireland",
        "cn" => "爱尔兰",
        "match" => /dublin|ireland|ucd/i
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
        "desc_cn" => "很多同学不是完全听不懂，而是教授口音重、语速快，专业词一闪就过，回去对着课件更对不上。" \
                     "同步跟课会按你学校的周进度，用中文把当周概念补清楚，同时保留 tutorial 和考试要用的英文说法，避免越掉越远。" \
                     "微信直约我本人，没有顾问转手。"
      },
      "assignment" => {
        "en" => "Assignment tutoring",
        "cn" => "作业辅导",
        "title_en" => "Assignment & Project Help",
        "title_cn" => "作业与 Project 辅导",
        "desc_en" => "Debug and approach review. Direct tutor, no middleman fee.",
        "desc_cn" => "作业卡壳往往卡在 brief 读不透、思路立不住，或本地能跑一提交就被边界情况打回来。" \
                     "我们会先一起读题、梳一条你能写下去的路，再对着报错现场排查；代码和推导仍由你完成提交，我负责把方法讲透。" \
                     "微信直约，无中介加价。"
      },
      "exam" => {
        "en" => "Exam prep tutoring",
        "cn" => "考前突击辅导",
        "title_en" => "Exam Prep Tutoring",
        "title_cn" => "考前突击辅导",
        "desc_en" => "Topic maps, past-paper style drills, weak-spot sprints before midterms and finals.",
        "desc_cn" => "临考最怕的不是「没刷够题」，而是不知道这门课反复考什么、自己弱在哪、下手顺序怎么排。" \
                     "期中期末前会先梳考点地图，再按常考题型和你的薄弱点做限时练习，练审题、表述和得分点，而不是无目的地刷题。"
      },
      "preview" => {
        "en" => "Course preview tutoring",
        "cn" => "课程预习辅导",
        "title_en" => "Course Preview Tutoring",
        "title_cn" => "课程预习辅导",
        "desc_en" => "Pre-term head start: programming basics, jargon, and assessment format before week 1.",
        "desc_cn" => "开学第一周最容易懵：课速突然加快，英文术语一串串砸过来，作业格式和评分点也还没摸清。" \
                     "开学前先把编程基础、专业说法和考核方式过一遍，后面跟课会轻松很多，不必从第二周就开始追进度。"
      }
    }.freeze

    # More specific themes first; "general" is the fallback (not listed here).
    # Theme packs live in _data/teach_themes/*.yml (admin: Course Themes).
    THEME_MATCH_ORDER = %w[
      machine-learning
      data-science
      security
      databases
      automata
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

      # Evidence catalog from .scripts/sync-teach-catalog.py (filenames/counts only).
      catalog = catalog_by_slug(site)

      universities = []
      slug_lookup = {}
      seen_slugs = {}
      rich_uni_payloads = []
      theme_index = Hash.new { |h, k| h[k] = [] }

      courses_data.each do |slug, entry|
        next unless entry.is_a?(Hash) && entry["university"]

        data_key = slug.to_s
        uni_slug = ascii_slug(data_key)
        next if uni_slug.empty?

        university = entry["university"].to_s.strip
        uni_abbr = entry["abbr"].to_s.strip
        uni_name_cn = entry["name_cn"].to_s.strip
        label_en, label_cn = uni_labels(university, uni_abbr, uni_name_cn)
        course_list = Array(entry["courses"]).map { |c| normalize_course(c, themes) }.compact
        next if course_list.empty?

        region_hay = [
          data_key, uni_slug, university, uni_abbr, uni_name_cn, entry["teaching_folder"]
        ].compact.join(" ")
        region_key = detect_region(region_hay, university)
        next if seen_slugs[uni_slug]

        evidence = catalog[uni_slug] || {}
        course_list = merge_course_evidence(course_list, evidence)

        rich_uni_payloads << {
          "uni_slug" => uni_slug,
          "university" => university,
          "uni_abbr" => uni_abbr,
          "uni_name_cn" => uni_name_cn,
          "label_en" => label_en,
          "label_cn" => label_cn,
          "course_list" => course_list,
          "region_key" => region_key,
          "evidence" => evidence
        }

        course_list.each do |course|
          theme_index[course["theme"]] << {
            "uni_slug" => uni_slug,
            "university" => university,
            "label_en" => label_en,
            "label_cn" => label_cn,
            "course" => course
          }
        end

        universities << {
          "slug" => uni_slug,
          "name" => university,
          "abbr" => uni_abbr,
          "name_cn" => uni_name_cn,
          "label_en" => label_en,
          "label_cn" => label_cn,
          "region" => region_key,
          "courses" => course_list,
          "rich" => true,
          "topics" => Array(evidence["topics"]),
          "signals" => evidence["signals"].is_a?(Hash) ? evidence["signals"] : {},
          "coverage_summary_en" => evidence["coverage_summary_en"].to_s,
          "coverage_summary_cn" => evidence["coverage_summary_cn"].to_s,
          "depth_score" => evidence["total_depth"].to_i
        }
        slug_lookup[data_key] = uni_slug
        seen_slugs[uni_slug] = true
      end

      rich_count = universities.size
      course_count = universities.sum { |u| u["courses"].size }
      site.data["teach_course_count"] = course_count

      reading_by_theme = {}
      themes.each do |theme_id, meta|
        next unless meta.is_a?(Hash)

        reading_by_theme[theme_id.to_s] = resolve_theme_reading(site, meta["reading"])
      end

      rich_uni_payloads.each do |payload|
        uni_slug = payload["uni_slug"]
        university = payload["university"]
        uni_abbr = payload["uni_abbr"]
        uni_name_cn = payload["uni_name_cn"]
        label_en = payload["label_en"]
        label_cn = payload["label_cn"]
        course_list = payload["course_list"]
        region_key = payload["region_key"]
        evidence = payload["evidence"]

        # Indexability gate (SEO combo): uni hub only when at least one course
        # has linked private materials; uni-service shells never index; course
        # pages index only when that course is linked.
        linked_n = evidence.is_a?(Hash) ? evidence["linked_course_count"].to_i : 0
        if linked_n <= 0
          linked_n = course_list.count { |c| c["linked"] }
        end
        uni_in_sitemap = linked_n >= 1

        site.pages << build_university_page(
          site, uni_slug, university, course_list, region_key, uni_in_sitemap, evidence,
          uni_abbr: uni_abbr, uni_name_cn: uni_name_cn, label_en: label_en, label_cn: label_cn
        )

        SERVICES.each_key do |service_key|
          site.pages << build_uni_service_page(
            site, uni_slug, university, course_list, region_key, service_key,
            false, evidence
          )
        end

        occupied_course_slugs = {}
        course_list.each do |course|
          course_slug = course["slug"].to_s
          if course_slug.empty? || occupied_course_slugs[course_slug]
            Jekyll.logger.warn "TeachPages:", "skipping duplicate course slug " \
              "#{uni_slug}/#{course_slug} (#{course['code']})" unless course_slug.empty?
            next
          end

          occupied_course_slugs[course_slug] = true
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
            siblings: course_list,
            evidence: evidence,
            uni_abbr: uni_abbr,
            uni_name_cn: uni_name_cn,
            label_en: label_en,
            label_cn: label_cn
          )
          emit_course_alias_redirects(site, uni_slug, course, occupied_course_slugs)
        end
      end

      universities.sort_by! { |u| [-u["depth_score"].to_i, u["name"].downcase] }
      site.data["teach_universities"] = universities
      site.data["teach_uni_slugs"] = slug_lookup
      site.data["teach_services"] = SERVICES
      site.data["teach_uni_count"] = universities.size
      site.data["teach_course_count"] = course_count
      # Keep bulk YAML off generated pages (legacy file may still exist on disk).
      site.data["teach_bulk_disabled"] = true

      REGION_MAP.each_key do |region_key|
        region_unis = universities.select { |u| u["region"] == region_key }
        next if region_unis.empty?

        site.pages << build_region_page(site, region_key, region_unis)
      end

      hub_unis = universities.select do |u|
        u["courses"].size.positive? &&
          %w[australia uk usa canada hong-kong singapore malaysia new-zealand].include?(u["region"])
      end
      SERVICES.each_key do |service_key|
        site.pages << build_service_hub_page(site, service_key, hub_unis)
      end

      site.pages << build_universities_index(site, universities)
      uni_service_count = universities.size * SERVICES.size
      theme_counts = theme_index.transform_values(&:size)
      Jekyll.logger.info "TeachPages:", "generated #{universities.size} evidence-backed universities " \
        "(dropped empty bulk directory), #{course_count} course pages, " \
        "#{uni_service_count} uni-service redirects→hub, #{SERVICES.size} service hubs, " \
        "#{REGION_MAP.count { |k, _| universities.any? { |u| u['region'] == k } }} regions; " \
        "themes=#{theme_counts.map { |k, v| "#{k}:#{v}" }.join(', ')}"
    end

    private

    def catalog_by_slug(site)
      raw = site.data["teach_catalog"]
      list = if raw.is_a?(Hash)
               Array(raw["universities"])
             elsif raw.is_a?(Array)
               raw
             else
               []
             end
      list.each_with_object({}) do |entry, acc|
        next unless entry.is_a?(Hash)

        slug = ascii_slug(entry["slug"].to_s)
        next if slug.empty?

        acc[slug] = entry
      end
    end

    # Must match confident how-strings from .scripts/sync-teach-catalog.py
    LINKED_HOW = [
      "material field",
      "material alias",
      "code match",
      "title fuzzy"
    ].freeze

    def merge_course_evidence(course_list, evidence)
      evidence_courses = Array(evidence["courses"])
      course_list.map do |course|
        match = find_evidence_course(course, evidence_courses)
        next course.merge("linked" => false) unless match

        how = match["link"].to_s
        linked = LINKED_HOW.include?(how) && match["depth_score"].to_i.positive?
        curated = Array(course["curated_topics"])
        course.merge(
          # Prefer hand-written focus topics over filename heuristics.
          "topics" => curated.any? ? curated : Array(match["topics"]),
          "signals" => match["signals"].is_a?(Hash) ? match["signals"] : {},
          "shape" => match["shape"].is_a?(Hash) ? match["shape"] : {},
          "depth_score" => match["depth_score"].to_i,
          "material" => match["material"].to_s,
          "link_how" => how,
          "linked" => linked
        )
      end
    end

    def find_evidence_course(course, evidence_courses)
      code_keys = course_code_keys(course)
      title_key = ascii_slug(course["title"].to_s)
      material_key = course["material"].to_s.strip
      evidence_courses.find do |ec|
        next false unless ec.is_a?(Hash)

        ec_code = ec["code"].to_s.gsub(/\s+/, "").upcase
        ec_title = ascii_slug(ec["title"].to_s)
        ec_material = ec["material"].to_s.strip
        (material_key != "" && ec_material == material_key) ||
          (ec_code != "" && code_keys.include?(ec_code)) ||
          (title_key != "" && (ec_title == title_key || ec_title.include?(title_key) || title_key.include?(ec_title)))
      end
    end

    def course_code_keys(course)
      ([course["code"]] + Array(course["aliases"])).map { |c| c.to_s.gsub(/\s+/, "").upcase }.reject(&:empty?).uniq
    end

    def bilingual_items(raw)
      Array(raw).select { |t| t.is_a?(Hash) && (t["en"].to_s.strip != "" || t["cn"].to_s.strip != "") }
    end

    # Per-theme difficulty hooks for Chinese students abroad (public copy, not inventory).
    # CN lines should read like a tutor talking to a student, not a keyword list.
    THEME_DIFFICULTY = {
      "automata" => {
        "en" => "Greek letters, abstract machines, and cryptic state / Petri diagrams",
        "cn" => "希腊字母一串串往下写，状态机和 Petri 网图看起来像暗号，课件翻完仍对不上作业到底在问哪一步"
      },
      "algorithms" => {
        "en" => "Proofs, complexity talk, and pseudocode under time pressure",
        "cn" => "证明要写严谨、复杂度一问就懵；限时伪代码往往不是语法不会，而是思路还没想清时间就没了"
      },
      "machine-learning" => {
        "en" => "Math+code lectures, loss jargon, and notebooks that fail on shape mismatches",
        "cn" => "课上公式和代码交错讲，loss、梯度听过却对不上 notebook；维度一错，报错信息又读不明白"
      },
      "data-science" => {
        "en" => "Assessment briefs, messy tables, and write-ups that must defend every plot",
        "cn" => "作业 brief 又长、表又脏，报告里每张图都要说得出为什么这样画；英文表述一卡，整段就写不动"
      },
      "programming" => {
        "en" => "Dense assignment wording, unfamiliar toolchains, and autograders that punish edge cases",
        "cn" => "作业说明密、工具链不熟，autograder 专挑边界情况扣分——本地能跑，交上去却红一片"
      },
      "systems" => {
        "en" => "Dense systems vocabulary, diagrams, and labs that fail for opaque reasons",
        "cn" => "系统术语又多又密，图一多就乱；Lab 挂了却分不清是环境、配置，还是概念本身没对齐"
      },
      "databases" => {
        "en" => "Schema / query jargon and assignments where one wrong join wrecks results",
        "cn" => "Schema 和查询说法搅在一起，一个 join 写偏后面结果全歪，改来改去仍不敢确定对不对"
      },
      "math-for-cs" => {
        "en" => "Proofs, symbol-heavy slides, and abstractions without a picture",
        "cn" => "证明步骤要求严，课件符号又密；很多抽象概念若没有画面，单靠背公式很难真正用起来"
      },
      "security" => {
        "en" => "Threat / crypto vocabulary and labs with cryptic failure modes",
        "cn" => "威胁模型和密码学术语叠在一起；Lab 一失败日志又含糊，很难判断是实现错了还是概念没对齐"
      },
      "web-mobile" => {
        "en" => "UI / API briefs and stacks where front-end and back-end errors blur together",
        "cn" => "UI 和 API 要求缠在一起，前后端报错互相掩盖，排查时常常分不清问题到底出在哪一层"
      },
      "general" => {
        "en" => "Fast lectures, dense terminology, and abstract ideas that do not stick from slides alone",
        "cn" => "课讲得快、术语密，很多抽象点单靠课件记不住，一到 tutorial 或作业就露馅"
      }
    }.freeze

    # Theme-specific CN lead openers (avoid one mechanical sentence for every course).
    THEME_LEAD_CN = {
      "automata" => "形式化这门课最折磨人的，往往不是定义背不下来，而是图和符号对不上作业在问的那一步。",
      "algorithms" => "算法课劝退，常常不是「完全没听过」，而是证明写不严谨、复杂度一问就空，限时题还没想清就结束了。",
      "machine-learning" => "机器学习课最容易卡在「词都听过、手却跟不上」：公式、代码、notebook 对不齐，维度一错整晚都在读报错。",
      "data-science" => "数据课的压力经常来自 brief 太长、表太脏，再加上报告要用英文把每张图的理由说圆。",
      "programming" => "入门和进阶编程课，难的往往不是语法本身，而是 brief 读不透、工具链不熟，以及 autograder 专砸边界情况。",
      "systems" => "系统课最怕「课能听个大概、Lab 却挂得莫名其妙」——术语密、图多，失败原因又不透明。",
      "databases" => "数据库作业看起来像在写查询，真正费时的是 Schema 关系没理清，一个 join 偏了后面全歪。",
      "math-for-cs" => "计算机数学课劝退，多半是符号太密、证明要求严，没有画面就只能死记，一到题目就不会用。",
      "security" => "安全课的门槛常常是术语墙：威胁模型、密码学说法叠在一起，Lab 失败时还分不清概念错还是实现错。",
      "web-mobile" => "前后端缠在同一份 brief 里时，最耗人的是排查——报错互相掩盖，分不清到底坏在哪一层。",
      "general" => "很多同学不是完全听不懂，而是课速一快、术语一密，笔记一断档，作业换个问法就不会下手。"
    }.freeze

    def theme_difficulty(theme_id)
      THEME_DIFFICULTY[theme_id.to_s] || THEME_DIFFICULTY["general"]
    end

    # Public course-page copy from theme knowledge when admin YAML has no hand-written fields.
    def synthesize_course_copy(course, university, theme_id, theme_meta, label_cn: nil)
      code = course["code"].to_s
      title = course["title"].to_s
      label = [code, title].reject(&:empty?).join(" ")
      label = title if label.empty?
      # Prefer short course handle in Chinese prose; dump full English title only when needed.
      cn_course = code.empty? ? title : code
      school = label_cn.to_s.strip
      school = university.to_s if school.empty?
      hard = theme_difficulty(theme_id)
      theme_label_en = theme_meta["label_en"].to_s
      theme_label_cn = theme_meta["label_cn"].to_s
      topic_ens = Array(theme_meta["topics"]).filter_map { |t| t["en"].to_s if t.is_a?(Hash) }.first(4)
      topic_cns = Array(theme_meta["topics"]).filter_map { |t|
        polish_cn(t["cn"]) if t.is_a?(Hash)
      }.first(3)

      lead_en = "#{label} at #{university} is hard for many Chinese students abroad: " \
                "#{hard['en']}. Tutoring turns that fog into steps you can repeat."
      opener = THEME_LEAD_CN[theme_id.to_s] || THEME_LEAD_CN["general"]
      lead_cn = polish_cn(
        "#{opener}在#{school}上 #{cn_course}，#{hard['cn']}。" \
        "一对一不会只给你更长的笔记，而是先把卡点拆开，按你跟得上的节奏把方法练熟，练到作业和考试能自己下手。"
      )

      overview_en = "#{label} sits in #{theme_label_en.to_s.downcase}. "
      overview_en += "Typical focus includes #{topic_ens.join('; ')}. " if topic_ens.any?
      overview_en += "The pain is rarely never having seen the topic; it is terminology " \
                     "under time pressure, symbols that do not stick, and questions that look " \
                     "cryptic until you have a method."

      overview_cn = "#{cn_course} 是 #{school}「#{theme_label_cn}」方向的课，考核通常既看你是否真懂，也看你能不能当场动手。"
      if topic_cns.any?
        overview_cn += "学期里常会绕着#{topic_cns.join('、')}这些点转。"
      end
      overview_cn += "留学生真正吃力的，往往不是「完全没见过」，而是课上术语来得太快、符号记不牢，" \
                     "作业和考试一换问法就不知道从哪下笔。" \
                     "辅导里会先帮你把方法立住，再对着你手里的题目练到能自己写、自己改。"
      overview_cn = polish_cn(overview_cn)

      stuck = Array(theme_meta["stuck"]).select { |t| t.is_a?(Hash) }.map do |t|
        t.merge("cn" => polish_cn(t["cn"]), "en" => t["en"].to_s)
      end
      pace_stuck = {
        "en" => "Lecture speed + dense #{theme_label_en.downcase} vocabulary outruns note-taking",
        "cn" => polish_cn("课速偏快，「#{theme_label_cn}」术语又密，笔记一跟丢，下课后更难自己补回来")
      }
      stuck = [pace_stuck] + stuck unless stuck.any? { |t|
        en = t["en"].to_s.downcase
        en.include?("lecture speed") || en.include?("outruns note")
      }

      session_angles = Array(theme_meta["session_angles"]).select { |t| t.is_a?(Hash) }.map do |t|
        t.merge("cn" => polish_cn(t["cn"]), "en" => t["en"].to_s)
      end
      tutoring = [
        {
          "en" => "Lecture sync for #{label}: bilingual catch-up so workshop / tutorial language matches the lectures.",
          "cn" => polish_cn(
            "按 #{cn_course} 当周进度跟课：用中文把课件里没听清的概念补上，" \
            "同时保留 workshop / tutorial 会用到的英文说法，避免听课和写作业变成两套语言。"
          )
        },
        {
          "en" => "Assignments: approach review and live debug.",
          "cn" => polish_cn(
            "作业卡壳时，先一起读透 brief 和评分点，梳一条你能写下去的思路；" \
            "你来写代码或推导，我在旁边盯卡点、一起看报错和边界，而不是代写交差。"
          )
        },
        {
          "en" => "Exam prep: timed drills on weak spots and recurring question shapes for this course.",
          "cn" => polish_cn(
            "期中期末前先摸清这门课常考的题型和你自己的薄弱点，再做限时练习；" \
            "练的是审题、下手顺序和表述，而不是临考前无目的地刷题。"
          )
        }
      ]

      {
        "lead_en" => lead_en,
        "lead_cn" => lead_cn,
        "overview_en" => overview_en,
        "overview_cn" => overview_cn,
        "stuck" => stuck,
        "session_angles" => session_angles,
        "tutoring" => tutoring,
        "topics" => Array(theme_meta["topics"]).select { |t| t.is_a?(Hash) }
      }
    end

    def normalize_course(course, themes = {})
      return nil unless course.is_a?(Hash)

      code = course["code"].to_s.strip
      title = course["title"].to_s.strip
      return nil if code.empty? && title.empty?

      aliases = normalize_aliases(course["aliases"], code)
      override = course["theme"].to_s.strip
      theme_id = if !override.empty? && themes.key?(override)
                   override
                 else
                   infer_theme("#{code} #{aliases.join(' ')} #{title}", themes)
                 end

      {
        "code" => code,
        "aliases" => aliases,
        "title" => title,
        "department" => course["department"].to_s.strip,
        "material" => course["material"].to_s.strip,
        "slug" => course_slug(code, title),
        "theme" => theme_id,
        "overview_en" => course["overview_en"].to_s.strip,
        "overview_cn" => course["overview_cn"].to_s.strip,
        "lead_en" => course["lead_en"].to_s.strip,
        "lead_cn" => course["lead_cn"].to_s.strip,
        "curated_topics" => bilingual_items(course["topics"]),
        "stuck" => bilingual_items(course["stuck"]),
        "session_angles" => bilingual_items(course["session_angles"]),
        "tutoring" => bilingual_items(course["tutoring"])
      }
    end

    # Accept Decap list-of-strings, list-of-objects, or a comma/semicolon string.
    def normalize_aliases(raw, primary_code)
      items = case raw
              when Array then raw
              when String then raw.split(/[,;|]/)
              else []
              end
      primary_norm = primary_code.to_s.gsub(/\s+/, "").upcase
      primary_slug = ascii_slug(primary_code)
      items.filter_map do |item|
        text = case item
               when Hash then (item["alias"] || item["code"] || item["name"]).to_s
               else item.to_s
               end
        text = text.strip
        next if text.empty?

        norm = text.gsub(/\s+/, "").upcase
        next if !primary_norm.empty? && norm == primary_norm
        next if !primary_slug.empty? && ascii_slug(text) == primary_slug

        text
      end.uniq
    end

    def emit_course_alias_redirects(site, uni_slug, course, occupied_course_slugs)
      canonical = course["slug"].to_s
      return if canonical.empty?

      target = "/teach/courses/#{uni_slug}/#{canonical}/"
      Array(course["aliases"]).each do |alias_code|
        alias_slug = ascii_slug(alias_code)
        next if alias_slug.empty? || alias_slug == canonical
        next if occupied_course_slugs[alias_slug]

        site.pages << TeachGeneratedPage.new(
          site,
          site.source,
          "teach/courses/#{uni_slug}/#{alias_slug}"
        ).tap do |page|
          page.data.merge!(
            "layout" => "redirect",
            "redirect_to" => target,
            "sitemap" => false,
            "robots" => "noindex, follow",
            "title" => "Redirecting…",
            "permalink" => "/teach/courses/#{uni_slug}/#{alias_slug}/"
          )
        end
        occupied_course_slugs[alias_slug] = true
      end
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

    # Course "Notes from my writing" is posts only — never glossary term pages.
    def resolve_theme_reading(site, reading_list)
      Array(reading_list).filter_map do |item|
        next unless item.is_a?(Hash)
        next unless item["kind"].to_s == "post"

        name = item["name"].to_s.strip
        next if name.empty?

        post = find_post(site, name)
        next unless post

        date = post.date
        {
          "url" => post.url,
          "title_en" => item["title_en"].to_s,
          "title_cn" => item["title_cn"].to_s,
          "date" => date ? date.strftime("%Y-%m-%d") : "",
          "date_label" => date ? date.strftime("%b %Y") : ""
        }
      end
    end

    # Free Matheion open lessons surfaced on matching course pages.
    MATHEION_JAVA = {
      "id" => "java",
      "url" => "https://matheion.zlu.me/learn/java/",
      "code" => "MA-JAVA0",
      "title_en" => "Java foundations",
      "title_cn" => "Java 基础"
    }.freeze

    MATHEION_NET = {
      "id" => "net",
      "url" => "https://matheion.zlu.me/learn/net/",
      "code" => "MA-NET1",
      "title_en" => "Computer networking",
      "title_cn" => "计算机网络"
    }.freeze

    def matheion_lessons_for_course(course, focus_topics = [])
      lessons = []
      lessons << MATHEION_JAVA if java_fundamental_course?(course, focus_topics)
      lessons << MATHEION_NET if networking_course?(course, focus_topics)
      lessons
    end

    # Intro / foundation Java (and typical Java OOP) course pages — not Python/C++/web.
    # Signals: code/title/material/aliases + hand-curated YAML topics only.
    def java_fundamental_course?(course, curated_topics = [])
      hay = [
        course["code"], course["title"], course["material"],
        *Array(course["aliases"])
      ].compact.join(" ")
      topic_hay = Array(curated_topics).filter_map { |t|
        next unless t.is_a?(Hash)

        "#{t['en']} #{t['cn']}"
      }.join(" ")
      text = "#{hay} #{topic_hay}"

      has_java = text.match?(/\bjava\b/i)
      other_lang = text.match?(
        /\b(python|c\+\+|javascript|typescript|web systems|data wrangling|foundations of computing)\b/i
      )
      return false if other_lang && !has_java
      return true if has_java
      return true if text.match?(/\bipro00[12]\b/i)

      # Classic OOP course titles in this catalog are usually Java.
      text.match?(/object[-\s]?oriented(\s+(programming|design|software(\s+development)?))?/i)
    end

    # Intro computer-networking courses (UTS INEF001, COMP9331, …) — not neural nets
    # or social/complex network analysis. Theme + title/material/code signals.
    def networking_course?(course, curated_topics = [])
      return true if course["theme"].to_s == "networking"

      hay = [
        course["code"], course["title"], course["material"],
        *Array(course["aliases"])
      ].compact.join(" ")
      topic_hay = Array(curated_topics).filter_map { |t|
        next unless t.is_a?(Hash)

        "#{t['en']} #{t['cn']}"
      }.join(" ")
      text = "#{hay} #{topic_hay}"

      return false if text.match?(
        /\b(neural|deep learning|social and information network|complex networks|pattern recognition)\b/i
      )
      return true if text.match?(/\b(inef001|comp9331|fit5124|inhn0012)\b/i)
      # Prefer plural-aware stems (Computer Networks, Systems and Networks, …).
      text.match?(
        /\b(
          computer\s+networks?|
          networking|
          network\s+fundamentals|
          networks?\s+and\s+(applications|communications|security)|
          (computer\s+)?systems?\s+and\s+networks?|
          architecture\s+and\s+networks?|
          wireless\s+networks?
        )\b/ix
      )
    end

    def find_post_url(site, name)
      find_post(site, name)&.url
    end

    def find_post(site, name)
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
      preferred || posts.first
    end

    def uni_labels(university, abbr, name_cn)
      university = university.to_s.strip
      abbr = abbr.to_s.strip
      name_cn = name_cn.to_s.strip
      label_en = if !abbr.empty? && !university.empty? && abbr.casecmp(university) != 0
                   "#{university} (#{abbr})"
                 else
                   (university.empty? ? abbr : university)
                 end
      label_cn = if !abbr.empty? && !name_cn.empty?
                   "#{abbr}（#{name_cn}）"
                 elsif !name_cn.empty?
                   name_cn
                 else
                   label_en
                 end
      [label_en, label_cn]
    end

    def related_course_payload(rec)
      course = rec["course"]
      {
        "uni_slug" => rec["uni_slug"],
        "university" => rec["university"],
        "label_en" => rec["label_en"],
        "label_cn" => rec["label_cn"],
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

    # Tighten CN prose: YAML folded newlines often leave spaces after ：。） etc.
    def polish_cn(text)
      text.to_s
          .gsub(/\s+/, " ")
          .gsub(/([：:;。；！？])\s+/, '\1')
          .gsub(/\s+([，。；！？、])/, '\1')
          .gsub(/([）』」])\s+/, '\1')
          .strip
    end

    # Baidu snippets prefer ~70–120 Chinese chars; cut on punctuation when possible.
    def truncate_meta(text, max = 120)
      s = polish_cn(text)
      return s if s.length <= max

      cut = s[0, max]
      idx = cut.rindex(/[。；;！？!?，,、\s]/)
      cut = cut[0, idx] if idx && idx > (max * 0.55)
      "#{cut.sub(/[。；;！？!?，,\s、]+\z/, '')}…"
    end

    # On-page + FAQPage items for course long-tail (作业 / 考前 / 英语).
    def course_faq_items(label_cn, code, title, tutoring, stuck)
      code_bit = code.to_s.strip
      code_bit = title.to_s.strip if code_bit.empty?
      code_bit = "本课" if code_bit.empty?
      stuck_cn = Array(stuck).first(2).filter_map { |t|
        polish_cn(t["cn"]) if t.is_a?(Hash)
      }.reject(&:empty?)
      stuck_note = if stuck_cn.any?
                     "这门课里同学特别容易卡在：#{stuck_cn.join('；')}。课上会优先对着这些点练，而不是泛泛重讲一遍课件。"
                   else
                     "课上会先问清你卡在读题、实现还是调试，再对着那一点往下推进。"
                   end
      [
        {
          "q" => "#{label_cn}#{code_bit}作业怎么辅导？能直接帮我写吗？",
          "a" => polish_cn(
            "不会代写交差。#{code_bit} 更常见的上法是：先一起把 brief、DDL 和评分点读透，梳一条你自己能写下去的思路，" \
            "再对着报错和边界情况现场排查；代码或推导仍由你完成提交。" \
            "这样考试换题时才不会只会看答案、自己仍下不了手。#{stuck_note}"
          )
        },
        {
          "q" => "#{code_bit}期末或期中临近，考前怎么突击更有效？",
          "a" => polish_cn(
            "临考最怕时间不够、题型不熟、下手顺序乱。不会带你无目的地刷题，而是先摸清 #{code_bit} 反复出现的考法和你自己的薄弱点，" \
            "再做限时练习：练审题、写步骤，以及答案该怎么组织才像在答教授的问题，尽量把有限的复习时间用在得分点上。"
          )
        },
        {
          "q" => "英语听力一般，跟得上这门课的辅导吗？",
          "a" => polish_cn(
            "跟得上。教授口音重、语速快、专业词一闪就过，本来就是很多同学来约 #{code_bit} 的原因。" \
            "课上会先用中文把直觉和推导讲透，再把作业、考试里会出现的英文术语对齐，" \
            "避免「中文听懂了、英文卷面仍写不出来」的断层。"
          )
        },
        {
          "q" => "想约 #{code_bit}，第一次怎么联系？",
          "a" => polish_cn(
            "微信直接发我即可：学校、课号（#{code_bit}）、截止日期，再加上题目 PDF、代码或报错截图。" \
            "资料越齐，第一节课就能对着你的卡点开讲，不必把半节课花在来回确认需求上。"
          )
        }
      ]
    end

    def build_university_page(
      site, uni_slug, university, courses, region_key, in_sitemap = true, evidence = {},
      uni_abbr: "", uni_name_cn: "", label_en: nil, label_cn: nil
    )
      evidence = {} unless evidence.is_a?(Hash)
      label_en, label_cn = uni_labels(university, uni_abbr, uni_name_cn) if label_en.nil? || label_cn.nil?
      # Hub list is A–Z by code; keep caller order for related-course picks.
      courses_sorted = Array(courses).sort_by do |c|
        [c["code"].to_s.downcase, c["title"].to_s.downcase]
      end
      codes = courses_sorted.flat_map { |c| [c["code"], *Array(c["aliases"])] }.reject(&:empty?).uniq
      title_en = "#{label_en} Computer Science Tutoring | AI & Programming"
      title_cn = "#{label_cn} 计算机辅导 | 人工智能与编程一对一"
      # Chinese meta for Baidu: scenario-led, not a keyword dump.
      code_bit = if codes.any?
                   "目录里已整理 #{codes.first(6).join('、')}#{codes.size > 6 ? ' 等课号' : ''}，"
                 else
                   "也可微信直接发课号咨询，"
                 end
      description = truncate_meta(
        "#{label_cn}计算机一对一辅导：课听不懂、作业卡壳、考前来不及，都可中英双语讲透。" \
        "#{code_bit}同步跟课、作业与考前突击均可约；咨询和上课是同一个人，无中介。"
      )
      en_codes = if codes.any?
                   "Courses: #{codes.first(8).join(', ')}#{codes.size > 8 ? '…' : ''}. "
                 else
                   ""
                 end
      description_en = truncate_meta(
        "Bilingual CS tutoring for #{label_en} students. #{en_codes}" \
        "Assignments, lecture sync, exam prep.",
        160
      )
      keyword_bits = [university, uni_abbr, uni_name_cn, "CS tutoring", "留学生辅导",
                      *codes.first(6), "计算机辅导"].reject { |x| x.to_s.strip.empty? }

      page = TeachGeneratedPage.new(site, site.source, "teach/universities/#{uni_slug}")
      page.data.merge!(
        "layout" => "teach_university",
        "title" => title_cn,
        "title_en" => title_en,
        "title_cn" => title_cn,
        "title_suffix" => "zlu.me/teach",
        "description" => description,
        "description_en" => description_en,
        "keywords" => keyword_bits.join(", "),
        "university" => university,
        "uni_abbr" => uni_abbr,
        "uni_name_cn" => uni_name_cn,
        "university_label_en" => label_en,
        "university_label_cn" => label_cn,
        "uni_slug" => uni_slug,
        "courses" => courses_sorted,
        "region" => region_key,
        "region_en" => region_label(region_key, "en"),
        "region_cn" => region_label(region_key, "cn"),
        "lang" => "zh-CN",
        "sitemap" => in_sitemap,
        "robots" => (in_sitemap ? nil : "noindex, follow"),
        # Hub page leads with the course list; skip template coverage copy.
        "evidence_topics" => [],
        "evidence_signals" => {},
        "coverage_summary_en" => "",
        "coverage_summary_cn" => "",
        "evidence_backed" => in_sitemap
      )
      page
    end

    def build_course_page(
      site, uni_slug, university, course, region_key,
      themes:, theme_index:, reading_by_theme:, course_count:, siblings:, evidence: {},
      uni_abbr: "", uni_name_cn: "", label_en: nil, label_cn: nil
    )
      label_en, label_cn = uni_labels(university, uni_abbr, uni_name_cn) if label_en.nil? || label_cn.nil?
      code = course["code"]
      aliases = Array(course["aliases"])
      title = course["title"]
      alias_bit = aliases.any? ? " (#{aliases.first})" : ""
      label = [code, title].reject(&:empty?).join(": ")
      theme_id = course["theme"].to_s
      theme_id = "general" if theme_id.empty?
      theme_meta = themes[theme_id]
      theme_meta = themes["general"] || {} unless theme_meta.is_a?(Hash)

      theme_label_en = theme_meta["label_en"].to_s
      theme_label_cn = theme_meta["label_cn"].to_s
      theme_topics = Array(theme_meta["topics"]).select { |t| t.is_a?(Hash) }
      course_topics = Array(course["topics"]).select { |t| t.is_a?(Hash) }
      course_signals = course["signals"].is_a?(Hash) ? course["signals"] : {}
      course_shape = course["shape"].is_a?(Hash) ? course["shape"] : {}

      hand_curated = [
        course["lead_en"], course["lead_cn"], course["overview_en"], course["overview_cn"]
      ].any? { |v| v.to_s.strip != "" }
      synth = synthesize_course_copy(
        course, university, theme_id, theme_meta, label_cn: label_cn
      )

      stuck = bilingual_items(course["stuck"])
      stuck = synth["stuck"] if stuck.empty?
      stuck = stuck.map { |t|
        next t unless t.is_a?(Hash)

        t.merge("cn" => polish_cn(t["cn"]))
      }
      session_angles = bilingual_items(course["session_angles"])
      session_angles = synth["session_angles"] if session_angles.empty?
      session_angles = session_angles.map { |t|
        next t unless t.is_a?(Hash)

        t.merge("cn" => polish_cn(t["cn"]))
      }
      lead_en = course["lead_en"].to_s
      lead_cn = polish_cn(course["lead_cn"])
      lead_en = synth["lead_en"] if lead_en.empty?
      lead_cn = synth["lead_cn"] if lead_cn.empty?
      overview_en = course["overview_en"].to_s
      overview_cn = polish_cn(course["overview_cn"])
      overview_en = synth["overview_en"] if overview_en.empty?
      overview_cn = synth["overview_cn"] if overview_cn.empty?
      tutoring = bilingual_items(course["tutoring"])
      tutoring = synth["tutoring"] if tutoring.empty?
      tutoring = tutoring.map { |t|
        next t unless t.is_a?(Hash)

        t.merge("cn" => polish_cn(t["cn"]))
      }

      # Prefer catalog/evidence topics, then hand-curated YAML topics, then theme topics.
      curated_topics = Array(course["curated_topics"]).select { |t| t.is_a?(Hash) }
      focus_topics = if course_topics.any?
                       course_topics
                     elsif curated_topics.any?
                       curated_topics
                     else
                       theme_topics
                     end
      topic_bits = focus_topics.first(4).map { |t| t["en"].to_s }.reject(&:empty?)
      topic_bits_cn = focus_topics.first(4).map { |t| t["cn"].to_s }.reject(&:empty?)

      # Bing/GSC SERP (AU course-code queries) via jekyll-seo.
      # Keep code + uni adjacent ("fit2102 monash"); CN titles stay for in-page lang switch.
      title_en = if code.empty?
                   "#{title} tutoring | #{label_en}"
                 else
                   "#{code} #{label_en} tutoring | #{title}#{alias_bit}"
                 end
      # Match competitor title pattern: 大学 + 课号 + 辅导
      title_cn = if code.empty?
                   "#{label_cn} #{title} 课程辅导 | 中英双语一对一"
                 else
                   "#{label_cn} #{code}#{alias_bit} 课程辅导 | #{title}"
                 end

      # CN snippet kept for Baidu / lang switch; EN is the indexed description.
      description_cn = if overview_cn.length >= 40
                         truncate_meta(overview_cn, 120)
                       else
                         alias_note = aliases.any? ? "（同课也写作 #{aliases.first}）" : ""
                         topic_note = topic_bits_cn.any? ? "常见会碰到#{topic_bits_cn.first(3).join('、')}等。" : ""
                         truncate_meta(
                           "#{label_cn}#{code.empty? ? '' : code}#{alias_note}一对一辅导。" \
                           "#{topic_note}课听不懂、作业卡壳、考前突击都可约；中英双语讲透，咨询和上课同一人。",
                           120
                         )
                       end
      # Always lead EN snippets with tutoring intent (never ship raw overview alone —
      # Bing/GSC show pos ~8–10 with ~1% CTR when the meta reads like a syllabus).
      serp_en = course["meta_description_en"].to_s.strip
      description_en = if !serp_en.empty?
                         truncate_meta(serp_en, 160)
                       else
                         head = if code.empty?
                                  "#{title} tutoring at #{label_en}"
                                else
                                  "#{code} tutoring at #{label_en} (#{title})"
                                end
                         head += " — also #{aliases.join(', ')}" if aliases.any?
                         stuck_en = Array(stuck).filter_map { |t|
                           t["en"].to_s.strip if t.is_a?(Hash)
                         }.reject(&:empty?).first.to_s
                         desc = "#{head}. "
                         if stuck_en.length.between?(18, 88)
                           hook = stuck_en.sub(/\A[[:upper:]]/) { |c| c.downcase }
                           desc += "Stuck on #{hook}? "
                         elsif topic_bits.any?
                           desc += "Focus: #{topic_bits.first(3).join('; ')}. "
                         end
                         desc += "Bilingual Chinese/English: lecture sync, assignments, exam prep."
                         truncate_meta(desc, 160)
                       end
      description = description_en

      course_faq = course_faq_items(label_cn, code, title, tutoring, stuck)

      topic_kw = focus_topics.first(6).flat_map { |t| [t["en"], t["cn"]] }.compact.map(&:to_s).reject(&:empty?)
      keywords = [
        code, *aliases, title, university, uni_abbr, uni_name_cn,
        theme_label_en, theme_label_cn,
        "作业辅导", "考前辅导", "同步辅导", "留学生",
        *topic_kw.first(8)
      ].reject { |x| x.to_s.strip.empty? }.join(", ")

      same_uni = related_same_uni(siblings, course, 4).map do |c|
        c.merge(
          "uni_slug" => uni_slug,
          "university" => university,
          "url" => "/teach/courses/#{uni_slug}/#{c['slug']}/"
        )
      end
      same_theme = related_same_theme(theme_index, theme_id, uni_slug, 6)
      theme_reading = Array(reading_by_theme[theme_id]).first(4)
      # Hand-curated YAML topics only — catalog evidence topics are filename heuristics
      # and can falsely tag web/Python courses with "Java".
      matheion_lessons = matheion_lessons_for_course(course, Array(course["curated_topics"]))

      about_topics = focus_topics.first(8).map { |t| t["en"].to_s }.reject(&:empty?)

      page = TeachGeneratedPage.new(
        site,
        site.source,
        "teach/courses/#{uni_slug}/#{course['slug']}"
      )
      page.data.merge!(
        "layout" => "teach_course",
        # Indexed title/description are English (GSC course-code queries).
        "title" => title_en,
        "title_en" => title_en,
        "title_cn" => title_cn,
        "title_suffix" => "zlu.me/teach",
        "description" => description,
        "description_en" => description_en,
        "description_cn" => description_cn,
        "keywords" => keywords,
        "university" => university,
        "uni_abbr" => uni_abbr,
        "uni_name_cn" => uni_name_cn,
        "university_label_en" => label_en,
        "university_label_cn" => label_cn,
        "uni_slug" => uni_slug,
        "course_code" => code,
        "course_aliases" => aliases,
        "course_title" => title,
        "course_department" => course["department"],
        "course_slug" => course["slug"],
        "theme" => theme_id,
        "theme_label_en" => theme_label_en,
        "theme_label_cn" => theme_label_cn,
        "theme_lead_en" => lead_en,
        "theme_lead_cn" => lead_cn,
        "overview_en" => overview_en,
        "overview_cn" => overview_cn,
        "theme_topics" => focus_topics,
        "theme_stuck" => stuck,
        "theme_session_angles" => session_angles,
        "tutoring" => tutoring,
        "course_faq" => course_faq,
        "theme_reading" => theme_reading,
        "matheion_lessons" => matheion_lessons,
        "theme_about" => about_topics,
        "related_same_uni" => same_uni,
        "related_same_theme" => same_theme,
        "teach_course_count" => course_count,
        "region" => region_key,
        "region_en" => region_label(region_key, "en"),
        "region_cn" => region_label(region_key, "cn"),
        "lang" => "zh-CN",
        # All course pages now carry theme/course copy — index them.
        "sitemap" => true,
        "robots" => nil,
        "evidence_topics" => course_topics,
        "evidence_signals" => course_signals,
        "evidence_shape" => course_shape,
        "evidence_backed" => !!course["linked"],
        "hand_curated" => hand_curated,
        "coverage_summary_en" => "",
        "coverage_summary_cn" => ""
      )
      page
    end

    # Thin /exam|/assignment|/sync|/preview shells attracted "free answers"
    # traffic and had no real content. Keep the URL as a soft redirect to the
    # uni hub (course list + tutoring copy). Never sitemap / IndexNow.
    def build_uni_service_page(
      site, uni_slug, _university, _courses, _region_key, service_key, _in_sitemap = false, _evidence = {}
    )
      target = "/teach/universities/#{uni_slug}/"
      page = TeachGeneratedPage.new(
        site,
        site.source,
        "teach/universities/#{uni_slug}/#{service_key}"
      )
      page.data.merge!(
        "layout" => "redirect",
        "redirect_to" => target,
        "title" => "Redirecting…",
        "permalink" => "/teach/universities/#{uni_slug}/#{service_key}/",
        "sitemap" => false,
        "robots" => "noindex, follow"
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
      description = truncate_meta(
        "留学生计算机#{svc['title_cn']}：#{svc['desc_cn']}" \
        "目前已整理 #{with_courses} 所高校的课程入口，覆盖澳英美加港新等地。",
        120
      )
      description_en = truncate_meta(
        "#{svc['desc_en']} #{with_courses}+ course-mapped universities; " \
        "#{ordered.size} schools across AU/UK/US/CA/HK/SG.",
        160
      )

      page = TeachGeneratedPage.new(site, site.source, "teach/services/#{service_key}")
      page.data.merge!(
        "layout" => "teach_service_hub",
        "title" => title_cn,
        "title_en" => title_en,
        "title_cn" => title_cn,
        "title_suffix" => "zlu.me/teach",
        "description" => description,
        "description_en" => description_en,
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
      description = truncate_meta(
        "#{cn}中国留学生的计算机一对一：教授口音重、课速快听不清，作业 brief 读不透，或考前来不及，" \
        "都可以用中英双语把概念讲透、对着题目练到手。本地区已整理 #{universities.size} 所高校、" \
        "#{course_count} 门课；微信发学校和课号即可约，咨询和上课是同一人。",
        120
      )
      description_en = truncate_meta(
        "Bilingual computer science tutoring for Chinese students in #{en}. " \
        "#{universities.size} universities · #{course_count}+ courses. WeChat booking.",
        160
      )

      page = TeachGeneratedPage.new(site, site.source, "teach/#{region_key}")
      page.data.merge!(
        "layout" => "teach_region",
        "title" => title_cn,
        "title_en" => title_en,
        "title_cn" => title_cn,
        "title_suffix" => "zlu.me/teach",
        "description" => description,
        "description_en" => description_en,
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
      title_en = "University CS Tutoring Directory"
      title_cn = "大学计算机辅导目录"
      course_n = universities.sum { |u| Array(u["courses"]).size }
      description = truncate_meta(
        "大学计算机辅导目录已整理 #{universities.size} 所高校、#{course_n} 门课。" \
        "按地区找到你的学校和课号后，可约同步跟课、作业与考前突击；中英双语，微信直约，无中介。",
        120
      )
      description_en = truncate_meta(
        "Directory of #{universities.size} universities and #{course_n} courses with " \
        "bilingual tutoring coverage. AI, data science, programming.",
        160
      )

      region_groups = []
      REGION_MAP.each do |key, meta|
        unis = universities.select { |u| u["region"] == key }
        next if unis.empty?

        region_groups << {
          "key" => key,
          "en" => meta["en"],
          "cn" => meta["cn"],
          "count" => unis.size,
          "course_count" => unis.sum { |u| Array(u["courses"]).size },
          "universities" => unis.sort_by { |u| u["name"].to_s.downcase }
        }
      end
      intl = universities.select { |u| u["region"] == "international" }
      if intl.any?
        region_groups << {
          "key" => "international",
          "en" => "International",
          "cn" => "海外 / 其他",
          "count" => intl.size,
          "course_count" => intl.sum { |u| Array(u["courses"]).size },
          "universities" => intl.sort_by { |u| u["name"].to_s.downcase }
        }
      end

      featured = universities
                   .select { |u| Array(u["courses"]).size.positive? }
                   .sort_by { |u| [-Array(u["courses"]).size, -u["depth_score"].to_i, u["name"].to_s.downcase] }
                   .first(8)

      page = TeachGeneratedPage.new(site, site.source, "teach/universities")
      page.data.merge!(
        "layout" => "teach_universities_index",
        "title" => title_cn,
        "title_en" => title_en,
        "title_cn" => title_cn,
        "title_suffix" => "zlu.me/teach",
        "description" => description,
        "description_en" => description_en,
        "keywords" => "university tutoring directory, 留学生辅导, computer science, AI",
        "universities" => universities,
        "region_groups" => region_groups,
        "featured" => featured,
        "course_count" => course_n,
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
