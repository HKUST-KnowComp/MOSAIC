const tabs = document.querySelectorAll(".tabs a");
const panels = document.querySelectorAll(".panel");

function show(name) {
  tabs.forEach((tab) => {
    const on = tab.dataset.tab === name;
    tab.setAttribute("aria-current", on ? "page" : "false");
  });
}

function fromHash() {
  const name = (location.hash || "#overview").slice(1);
  show(panels.some((panel) => panel.dataset.panel === name) ? name : "overview");
}

window.addEventListener("hashchange", fromHash);
fromHash();

document.getElementById("submit-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const data = new FormData(event.currentTarget);
  const body = [
    "### Result submission",
    "",
    `- System: ${data.get("system")}`,
    `- Track: ${data.get("track")}`,
    `- Memory: ${data.get("representation")}`,
    `- Reader: ${data.get("reader")}`,
    `- Answerable score: ${data.get("score")}`,
    `- Correct refusal: ${data.get("correct") || "n/a"}`,
    `- False refusal: ${data.get("false_refusal") || "n/a"}`,
    `- URL: ${data.get("url") || "n/a"}`,
    "",
    String(data.get("notes") || ""),
    "",
    "After a check, add the row to `docs/leaderboard.json`.",
  ].join("\n");
  const url = new URL("https://github.com/HKUST-KnowComp/MOSAIC/issues/new");
  url.searchParams.set("title", `Result submission: ${data.get("system")}`);
  url.searchParams.set("body", body);
  window.open(url.toString(), "_blank", "noopener");
});
