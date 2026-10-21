const tabs = document.querySelectorAll(".tabs a");
const panels = document.querySelectorAll(".panel");

function show(name) {
  panels.forEach((panel) => {
    panel.hidden = panel.dataset.panel !== name;
  });
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

const board = document.getElementById("board");
const boardNote = document.getElementById("board-note");
const boardStatus = document.getElementById("board-status");
const trackButtons = document.querySelectorAll(".track-switch button");
let leaderboard = null;
let activeTrack = "b";
let sortKey = null;
let sortLower = false;

function formatCell(column, value) {
  if (value === null || value === undefined || value === "") return "—";
  if (!column.numeric) return String(value);
  return Number(value).toFixed(column.digits);
}

function sortedRows(track) {
  const column = track.columns.find((item) => item.key === sortKey);
  const rows = track.rows.slice();
  if (!column || !column.numeric) return rows;
  const direction = column.lower ? 1 : -1;
  rows.sort((a, b) => {
    const av = a[sortKey];
    const bv = b[sortKey];
    if (av === null || av === undefined) return 1;
    if (bv === null || bv === undefined) return -1;
    return (av - bv) * direction;
  });
  return rows;
}

function bestValue(rows, column) {
  const values = rows.map((row) => row[column.key]).filter((value) => value !== null && value !== undefined);
  if (!values.length) return null;
  return column.lower ? Math.min(...values) : Math.max(...values);
}

function renderBoard() {
  const track = leaderboard.tracks.find((item) => item.id === activeTrack);
  if (!sortKey) sortKey = track.sort;
  const rows = sortedRows(track);
  boardNote.textContent = track.note;
  const head = document.createElement("tr");
  const rankHead = document.createElement("th");
  rankHead.textContent = "#";
  head.append(rankHead);
  track.columns.forEach((column) => {
    const cell = document.createElement("th");
    cell.textContent = column.label + (column.key === sortKey ? (column.lower ? " ↑" : " ↓") : "");
    cell.scope = "col";
    if (column.numeric) cell.className = "num";
    cell.addEventListener("click", () => {
      sortKey = column.key;
      sortLower = Boolean(column.lower);
      renderBoard();
    });
    head.append(cell);
  });
  board.querySelector("thead").replaceChildren(head);

  const body = document.createElement("tbody");
  rows.forEach((row, index) => {
    const tr = document.createElement("tr");
    const rank = document.createElement("td");
    rank.textContent = String(index + 1);
    tr.append(rank);
    track.columns.forEach((column) => {
      const cell = document.createElement("td");
      if (column.numeric) cell.className = "num";
      cell.textContent = formatCell(column, row[column.key]);
      if (column.numeric && row[column.key] !== null && row[column.key] === bestValue(rows, column)) {
        cell.classList.add("best");
      }
      if (column.key === "system" && row.source) {
        const badge = document.createElement("span");
        badge.className = "badge";
        badge.textContent = row.source;
        cell.append(badge);
      }
      tr.append(cell);
    });
    body.append(tr);
  });
  board.querySelector("tbody").replaceWith(body);
  boardStatus.textContent = `${rows.length} rows · paper baselines from the released evaluation.`;
  void sortLower;
}

trackButtons.forEach((button) => {
  button.addEventListener("click", () => {
    activeTrack = button.dataset.track;
    const track = leaderboard.tracks.find((item) => item.id === activeTrack);
    sortKey = track.sort;
    trackButtons.forEach((item) => {
      item.setAttribute("aria-selected", item === button ? "true" : "false");
    });
    renderBoard();
  });
});

fetch("leaderboard.json")
  .then((response) => {
    if (!response.ok) throw new Error(String(response.status));
    return response.json();
  })
  .then((data) => {
    leaderboard = data;
    renderBoard();
  })
  .catch(() => {
    boardStatus.textContent = "The leaderboard file did not load. Open this page from the project site, not as a loose file.";
  });
