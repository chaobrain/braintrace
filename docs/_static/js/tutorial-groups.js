(() => {
  "use strict";

  const groupSpecs = [
    {
      title: "Training workflows",
      documents: ["rnn_online_learning.html", "snn_online_learning.html"],
    },
    {
      title: "Algorithm tutorials",
      documents: ["drtrl.html", "pp_prop.html"],
    },
    {
      title: "Compiler & runtime",
      documents: [
        "customizing_primitive_transforms.html",
        "hidden_states.html",
        "graph_visualization.html",
        "batching.html",
      ],
    },
  ];

  function filenameFromUrl(url) {
    const pathname = new URL(url, window.location.href).pathname;
    return decodeURIComponent(pathname.split("/").pop() || "index.html");
  }

  function documentFilename(link) {
    const href = link.getAttribute("href");
    if (!href) return null;
    if (href === "#") {
      const currentFilename = filenameFromUrl(window.location.href);
      link.setAttribute("href", currentFilename);
      return currentFilename;
    }
    return filenameFromUrl(href);
  }

  function containsCurrentPage(items) {
    return items.some(
      (item) =>
        item.classList.contains("current") ||
        item.querySelector(".current, [aria-current='page']")
    );
  }

  function groupTutorialNavigation() {
    const tutorialCaption = Array.from(
      document.querySelectorAll(".bd-sidebar-primary .caption-text")
    ).find((caption) => caption.textContent.trim() === "Tutorial");
    if (!tutorialCaption) return;

    const caption = tutorialCaption.closest(".caption");
    const tutorialList = caption && caption.nextElementSibling;
    if (!tutorialList || !tutorialList.classList.contains("bd-sidenav")) return;
    if (tutorialList.dataset.braintraceGrouped === "true") return;

    const records = Array.from(tutorialList.children)
      .filter((item) => item.matches("li.toctree-l1"))
      .map((item) => {
        const link = item.querySelector(":scope > a[href]");
        return { item, filename: link ? documentFilename(link) : null };
      });

    groupSpecs.forEach((spec, index) => {
      const expected = new Set(spec.documents);
      const members = records
        .filter((record) => expected.has(record.filename))
        .map((record) => record.item);
      if (members.length !== spec.documents.length) return;

      const wrapper = document.createElement("li");
      const button = document.createElement("button");
      const childList = document.createElement("ul");
      const groupId = `braintrace-tutorial-group-${index + 1}`;
      const expanded = containsCurrentPage(members);

      wrapper.className = "braintrace-tutorial-group";
      button.className = "braintrace-tutorial-group-toggle";
      button.type = "button";
      button.textContent = spec.title;
      button.setAttribute("aria-controls", groupId);
      button.setAttribute("aria-expanded", String(expanded));
      childList.className = "braintrace-tutorial-group-list";
      childList.id = groupId;
      childList.hidden = !expanded;

      button.addEventListener("click", () => {
        const nextExpanded = button.getAttribute("aria-expanded") !== "true";
        button.setAttribute("aria-expanded", String(nextExpanded));
        childList.hidden = !nextExpanded;
      });

      tutorialList.insertBefore(wrapper, members[0]);
      wrapper.append(button, childList);
      members.forEach((item) => childList.appendChild(item));
    });

    tutorialList.dataset.braintraceGrouped = "true";
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", groupTutorialNavigation);
  } else {
    groupTutorialNavigation();
  }
})();
