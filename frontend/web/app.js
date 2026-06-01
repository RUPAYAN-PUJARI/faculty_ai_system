const { useState } = React;

function App() {
  const [prompt, setPrompt] = useState("");
  const [imageFile, setImageFile] = useState(null);
  const [status, setStatus] = useState("");
  const [messages, setMessages] = useState([]);
  const [lastContextId, setLastContextId] = useState(null);
  const [expandedScholarships, setExpandedScholarships] = useState({});

  const pushMessage = (role, content) => {
    setMessages((prev) => [...prev, { role, content }]);
  };

  const pushAssistantMessage = (content, results, meta = null) => {
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content, results: results || [], meta },
    ]);
  };
  const handleSubmit = async () => {
    const trimmedPrompt = prompt.trim();
    if (!trimmedPrompt && !imageFile) {
      setStatus("Type a prompt or add a photo.");
      return;
    }

    setStatus("Asking AI...");
    setPrompt("");

    if (imageFile) {
      pushMessage("user", trimmedPrompt || "Identify this faculty member");
      try {
        const formData = new FormData();
        formData.append("image", imageFile);
        if (trimmedPrompt) {
          formData.append("prompt", trimmedPrompt);
        }
        const response = await fetch("/api/assistant/image", {
          method: "POST",
          body: formData,
        });
        const data = await response.json();
        if (!response.ok) {
          setStatus(data?.error || "Assistant failed.");
          pushAssistantMessage(data?.error || "Assistant failed.", []);
          return;
        }
        setStatus("");
        const imageResults = Array.isArray(data?.results) ? data.results : [];
        if (imageResults.length > 0) {
          setLastContextId(imageResults[0].id || null);
        }
        pushAssistantMessage(data?.answer || "Done.", imageResults);
      } catch (error) {
        setStatus("Assistant failed.");
        pushAssistantMessage("Assistant failed.", []);
      }
      return;
    }

    pushMessage("user", trimmedPrompt);
    try {
      const response = await fetch("/api/assistant/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: trimmedPrompt,
          context_id: lastContextId || undefined,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        setStatus(data?.error || "Assistant failed.");
        pushAssistantMessage(data?.error || "Assistant failed.", []);
        return;
      }
      setStatus("");
      const responseType = data && data.result_type;
      const textResults = Array.isArray(data?.results) ? data.results : [];
      const normalizedResults = responseType
        ? textResults.map((item) => {
            if (item && typeof item === "object" && !Array.isArray(item)) {
              return item.result_type
                ? item
                : { ...item, result_type: responseType };
            }
            return item;
          })
        : textResults;
      const meta =
        responseType === "placement"
          ? {
              type: "placement",
              prompt: trimmedPrompt,
              nextOffset: data?.next_offset ?? null,
              offset: data?.offset ?? 0,
              limit: data?.limit ?? 50,
              total: data?.total ?? normalizedResults.length,
            }
          : null;
      if (normalizedResults.length > 0) {
        setLastContextId(normalizedResults[0].id || null);
      }
      pushAssistantMessage(data?.answer || "Done.", normalizedResults, meta);
    } catch (error) {
      setStatus("Assistant failed.");
      pushAssistantMessage("Assistant failed.", []);
    }
  };

  const loadMorePlacements = async (meta) => {
    if (!meta?.nextOffset) {
      return;
    }
    setStatus("Loading more placements...");
    try {
      const response = await fetch("/api/assistant/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: meta.prompt,
          placement_offset: meta.nextOffset,
          placement_limit: meta.limit || 50,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        setStatus(data?.error || "Assistant failed.");
        pushAssistantMessage(data?.error || "Assistant failed.", []);
        return;
      }
      setStatus("");
      const responseType = data && data.result_type;
      const textResults = Array.isArray(data?.results) ? data.results : [];
      const normalizedResults = responseType
        ? textResults.map((item) => {
            if (item && typeof item === "object" && !Array.isArray(item)) {
              return item.result_type
                ? item
                : { ...item, result_type: responseType };
            }
            return item;
          })
        : textResults;
      const newMeta =
        responseType === "placement"
          ? {
              type: "placement",
              prompt: meta.prompt,
              nextOffset: data?.next_offset ?? null,
              offset: data?.offset ?? 0,
              limit: data?.limit ?? 50,
              total: data?.total ?? normalizedResults.length,
            }
          : null;
      pushAssistantMessage(data?.answer || "Done.", normalizedResults, newMeta);
    } catch (error) {
      setStatus("Assistant failed.");
      pushAssistantMessage("Assistant failed.", []);
    }
  };

  const clearImage = () => {
    setImageFile(null);
  };

  const clearChat = () => {
    setMessages([]);
    setStatus("");
    setLastContextId(null);
  };

  const getResultType = (item) => {
    if (item && typeof item === "object") {
      if (item.result_type) {
        return item.result_type;
      }
      if (item.url || item.pdf_links || item.source === "bppimt_web") {
        return "web";
      }
      if (item.student_name || item.enrollment_number || item.employer) {
        return "placement";
      }
      if (
        item.scholarship_type ||
        item.official_url ||
        item.categories ||
        item.target_groups
      ) {
        return "scholarship";
      }
    }
    return "faculty";
  };

  const toggleScholarship = (key) => {
    setExpandedScholarships((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  const renderPlacementTable = (items) =>
    React.createElement(
      "div",
      { className: "placement-table-wrap" },
      React.createElement(
        "table",
        { className: "placement-table" },
        React.createElement(
          "thead",
          null,
          React.createElement(
            "tr",
            null,
            React.createElement("th", null, "Student"),
            React.createElement("th", null, "Discipline"),
            React.createElement("th", null, "Year"),
            React.createElement("th", null, "Campus"),
            React.createElement("th", null, "Employer"),
            React.createElement("th", null, "Academic Year"),
            React.createElement("th", null, "Enrollment"),
          ),
        ),
        React.createElement(
          "tbody",
          null,
          items.map((item, rowIdx) =>
            React.createElement(
              "tr",
              {
                key: `${item.enrollment_number || item.student_name}-${rowIdx}`,
              },
              React.createElement("td", null, item.student_name || "-"),
              React.createElement("td", null, item.discipline || "-"),
              React.createElement("td", null, item.year_of_passing || "-"),
              React.createElement(
                "td",
                null,
                item.on_off_campus ? `${item.on_off_campus} campus` : "-",
              ),
              React.createElement("td", null, item.employer || "-"),
              React.createElement("td", null, item.academic_year || "-"),
              React.createElement("td", null, item.enrollment_number || "-"),
            ),
          ),
        ),
      ),
    );

  const renderScholarshipCard = (item, itemIdx) => {
    const title = item.title || item.short_name || "Scholarship";
    const scholarshipKey = item.id || `${title}-${itemIdx}`;
    const expanded = !!expandedScholarships[scholarshipKey];
    const imageSrc = item.id ? `/scholarship-images/${item.id}.png` : "";
    const metaItems = [
      item.scholarship_type ? `Type: ${item.scholarship_type}` : "",
      item.state ? `State: ${item.state}` : "",
    ].filter(Boolean);

    const detailRows = [];
    if (item.categories?.length) {
      detailRows.push({
        label: "Categories",
        value: item.categories.join(", "),
      });
    }
    if (item.target_groups?.length) {
      detailRows.push({
        label: "Target groups",
        value: item.target_groups.join(", "),
      });
    }
    if (item.eligible_communities?.length) {
      detailRows.push({
        label: "Eligible communities",
        value: item.eligible_communities.join(", "),
      });
    }
    if (item.eligible_courses?.length) {
      detailRows.push({
        label: "Eligible courses",
        value: item.eligible_courses.join(", "),
      });
    }
    if (item.application_mode) {
      detailRows.push({
        label: "Application mode",
        value: item.application_mode,
      });
    }
    if (item.keywords?.length) {
      detailRows.push({ label: "Keywords", value: item.keywords.join(", ") });
    }
    if (item.special_conditions?.length) {
      detailRows.push({
        label: "Special conditions",
        value: item.special_conditions.join(", "),
      });
    }

    const eligibilityEntries =
      item.eligibility && typeof item.eligibility === "object"
        ? Object.entries(item.eligibility)
        : [];
    const benefitsEntries =
      item.benefits && typeof item.benefits === "object"
        ? Object.entries(item.benefits)
        : [];

    return React.createElement(
      "div",
      {
        className: "scholarship-card",
        key: `${item.id || title}-${itemIdx}`,
      },
      React.createElement(
        "div",
        { className: "scholarship-header-row" },
        React.createElement(
          "div",
          { className: "scholarship-media" },
          imageSrc
            ? React.createElement("img", {
                src: imageSrc,
                alt: `${title} logo`,
                onError: (event) => {
                  event.currentTarget.style.display = "none";
                },
              })
            : React.createElement(
                "div",
                { className: "scholarship-placeholder" },
                "No Image",
              ),
        ),
        React.createElement(
          "div",
          { className: "scholarship-content" },
          React.createElement(
            "div",
            { className: "scholarship-title-row" },
            React.createElement("h3", null, title),
            React.createElement(
              "span",
              { className: "scholarship-badge" },
              React.createElement("i", { className: "bi bi-award-fill" }),
              "Scholarship",
            ),
          ),
          metaItems.length
            ? React.createElement(
                "div",
                { className: "scholarship-meta" },
                metaItems.map((metaItem, metaIdx) =>
                  React.createElement(
                    "span",
                    {
                      className: "result-chip",
                      key: `meta-${itemIdx}-${metaIdx}`,
                    },
                    metaItem,
                  ),
                ),
              )
            : null,
          item.provider
            ? React.createElement(
                "p",
                { className: "scholarship-provider" },
                `Provider: ${item.provider}`,
              )
            : null,
          item.overview
            ? React.createElement(
                "p",
                { className: "scholarship-overview" },
                item.overview,
              )
            : null,
          React.createElement(
            "button",
            {
              className: "scholarship-toggle",
              onClick: () => toggleScholarship(scholarshipKey),
              type: "button",
            },
            expanded ? "Show less" : "Read more",
          ),
        ),
      ),
      expanded
        ? React.createElement(
            "div",
            { className: "scholarship-details" },
            detailRows.map((row, rowIdx) =>
              React.createElement(
                "div",
                { className: "detail-row", key: `detail-${rowIdx}` },
                React.createElement(
                  "span",
                  { className: "detail-label" },
                  row.label,
                ),
                React.createElement(
                  "span",
                  { className: "detail-value" },
                  row.value,
                ),
              ),
            ),
            eligibilityEntries.length
              ? React.createElement(
                  "div",
                  { className: "detail-block" },
                  React.createElement(
                    "span",
                    { className: "detail-label" },
                    "Eligibility",
                  ),
                  React.createElement(
                    "ul",
                    { className: "detail-list" },
                    eligibilityEntries.map(([key, value]) =>
                      React.createElement(
                        "li",
                        { key: `eligibility-${key}` },
                        `${key.replace(/_/g, " ")}: ${value}`,
                      ),
                    ),
                  ),
                )
              : null,
            benefitsEntries.length
              ? React.createElement(
                  "div",
                  { className: "detail-block" },
                  React.createElement(
                    "span",
                    { className: "detail-label" },
                    "Benefits",
                  ),
                  React.createElement(
                    "ul",
                    { className: "detail-list" },
                    benefitsEntries.map(([key, value]) =>
                      React.createElement(
                        "li",
                        { key: `benefits-${key}` },
                        `${key.replace(/_/g, " ")}: ${value}`,
                      ),
                    ),
                  ),
                )
              : null,
            item.official_url
              ? React.createElement(
                  "a",
                  {
                    href: item.official_url,
                    target: "_blank",
                    rel: "noreferrer",
                    className: "result-link",
                  },
                  "Official link",
                )
              : null,
          )
        : null,
    );
  };

  const renderWebCard = (item, itemIdx) => {
    const title = item.title || item.url || "BPPIMT Website";
    const pdfLinks = Array.isArray(item.pdf_links) ? item.pdf_links : [];

    return React.createElement(
      "div",
      {
        className: "web-card",
        key: `web-${itemIdx}`,
        style: { overflow: "hidden", minWidth: 0 }, // ← contain the card
      },
      React.createElement(
        "div",
        { className: "web-card-header" },
        React.createElement(
          "span",
          { className: "result-chip" },
          "BPPIMT Site",
        ),
      ),
      React.createElement("h3", null, title),
      item.snippet
        ? React.createElement(
            "p",
            {
              className: "web-snippet",
              style: { overflowWrap: "break-word", wordBreak: "break-word" }, // ← fix snippet
            },
            item.snippet,
          )
        : null,
      item.url
        ? React.createElement(
            "a",
            {
              href: item.url,
              target: "_blank",
              rel: "noreferrer",
              className: "result-link",
            },
            "Open page",
          )
        : null,
      pdfLinks.length
        ? React.createElement(
            "div",
            { className: "web-links" },
            pdfLinks.map((link, linkIdx) =>
              React.createElement(
                "a",
                {
                  key: `web-pdf-${itemIdx}-${linkIdx}`,
                  href: link.url,
                  target: "_blank",
                  rel: "noreferrer",
                  className: "result-link",
                  style: { wordBreak: "break-all", display: "block" }, // ← fix raw URLs
                },
                link.title || "PDF",
              ),
            ),
          )
        : null,
    );
  };

  return React.createElement(
    "div",
    { className: "container" },
    React.createElement(
      "header",
      null,
      React.createElement(
        "div",
        { className: "headline" },
        React.createElement("img", {
          className: "logo",
          src: "/images/logo.svg",
          alt: "BPPIMT logo",
        }),
        React.createElement(
          "div",
          null,
          React.createElement("h1", null, "BPPIMT Smart AI Search Assistant"),
          React.createElement(
            "p",
            { className: "subtitle" },
            "Find faculty profiles via text or photo in seconds",
          ),
        ),
      ),
      React.createElement(
        "div",
        { className: "header-actions" },
        React.createElement(
          "div",
          { className: "tag" },
          React.createElement("i", { className: "bi bi-stars" }),
          "Smart Retrieval",
        ),
      ),
    ),
    React.createElement(
      "div",
      { className: "chat-shell" },
      React.createElement(
        "div",
        { className: "chat-panel" },
        messages.length === 0
          ? React.createElement(
              "div",
              { className: "empty-chat" },
              "Ask a question or add a photo to start.",
            )
          : null,
        messages.map((msg, idx) =>
          React.createElement(
            "div",
            {
              key: `${msg.role}-${idx}`,
              className: `chat-message ${msg.role}`,
            },
            React.createElement(
              "div",
              { className: "bubble" },
              msg.content,
              msg.role === "assistant" && msg.results?.length
                ? React.createElement(
                    "div",
                    { className: "chat-results" },
                    (() => {
                      const results = msg.results || [];
                      const placements = results.filter(
                        (item) => getResultType(item) === "placement",
                      );
                      const scholarships = results.filter(
                        (item) => getResultType(item) === "scholarship",
                      );
                      const webItems = results.filter(
                        (item) => getResultType(item) === "web",
                      );
                      const facultyItems = results.filter(
                        (item) => getResultType(item) === "faculty",
                      );

                      return [
                        placements.length
                          ? React.createElement(
                              "div",
                              { key: "placement-table" },
                              renderPlacementTable(placements),
                              msg.meta?.type === "placement" &&
                                msg.meta?.nextOffset
                                ? React.createElement(
                                    "button",
                                    {
                                      className: "placement-more",
                                      onClick: () =>
                                        loadMorePlacements(msg.meta),
                                      type: "button",
                                    },
                                    "Load more placements",
                                  )
                                : null,
                            )
                          : null,
                        scholarships.length
                          ? React.createElement(
                              "div",
                              {
                                className: "scholarship-grid",
                                key: "scholarship-cards",
                              },
                              scholarships.map((item, itemIdx) =>
                                renderScholarshipCard(item, itemIdx),
                              ),
                            )
                          : null,
                        webItems.length
                          ? React.createElement(
                              "div",
                              { className: "web-grid", key: "web-cards" },
                              webItems.map((item, itemIdx) =>
                                renderWebCard(item, itemIdx),
                              ),
                            )
                          : null,
                        facultyItems.map((item, itemIdx) => {
                          const title = item.name || "Faculty Profile";
                          const meta = [
                            item.present_designation,
                            item.department,
                          ]
                            .filter(Boolean)
                            .join(" • ");
                          return React.createElement(
                            "div",
                            {
                              className: "result",
                              key: `${item.id || title}-${itemIdx}`,
                            },
                            React.createElement(
                              "div",
                              null,
                              item.image_url
                                ? React.createElement("img", {
                                    src: item.image_url,
                                    alt: item.name || "Faculty photo",
                                  })
                                : React.createElement(
                                    "div",
                                    { className: "tag" },
                                    "No Image",
                                  ),
                            ),
                            React.createElement(
                              "div",
                              null,
                              React.createElement("h3", null, title),
                              meta
                                ? React.createElement("p", null, meta)
                                : null,
                              React.createElement(
                                "p",
                                null,
                                item.specialization
                                  ? `Specialization: ${item.specialization}`
                                  : "",
                              ),
                            ),
                          );
                        }),
                      ];
                    })(),
                  )
                : null,
            ),
          ),
        ),
      ),
      React.createElement(
        "div",
        { className: "chat-side" },

        React.createElement(
          "div",
          { className: "side-fixed" },

          React.createElement(
            "div",
            { className: "card" },
            React.createElement("h2", null, "Tips"),
            React.createElement(
              "ul",
              { className: "tips" },
              React.createElement(
                "li",
                null,
                'Try: "Show CSE assistant professors"',
              ),
              React.createElement(
                "li",
                null,
                "Upload a photo and press Search to identify",
              ),
              React.createElement(
                "li",
                null,
                "Ask about designations, departments, or experience",
              ),
            ),
          ),

          React.createElement(
            "button",
            {
              className: "clear-chat fixed-clear",
              onClick: clearChat,
            },
            React.createElement("i", { className: "bi bi-trash" }),
            "Clear chat",
          ),
        ),
      ),
    ),
    React.createElement(
      "div",
      { className: "search-bar" },
      React.createElement(
        "div",
        { className: "search-input" },
        React.createElement("i", { className: "bi bi-search" }),
        imageFile
          ? React.createElement(
              "span",
              { className: "search-file" },
              React.createElement("span", null, imageFile.name),
              React.createElement(
                "button",
                { className: "chip-close", onClick: clearImage },
                React.createElement("i", { className: "bi bi-x-lg" }),
              ),
            )
          : null,
        React.createElement("input", {
          type: "text",
          value: prompt,
          onChange: (e) => setPrompt(e.target.value),
          placeholder: "Ask about faculty, departments, or upload a photo",
        }),
      ),
      React.createElement(
        "label",
        { className: "upload-btn" },
        React.createElement("span", { className: "upload-plus" }, "+"),
        React.createElement("input", {
          type: "file",
          accept: "image/*",
          onChange: (e) => setImageFile(e.target.files[0]),
        }),
      ),
      React.createElement(
        "button",
        { className: "primary", onClick: handleSubmit },
        React.createElement("i", { className: "bi bi-send" }),
        "Search",
      ),
    ),
    status ? React.createElement("p", { className: "status" }, status) : null,
    null,
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(React.createElement(App));
