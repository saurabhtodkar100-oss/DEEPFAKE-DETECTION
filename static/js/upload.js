const input = document.getElementById("mediaInput");
const dropZone = document.getElementById("dropZone");
const fileLine = document.getElementById("fileLine");
const uploadForm = document.getElementById("uploadForm");
const submitButton = uploadForm?.querySelector("button[type='submit']");

const formatBytes = (size) => {
  if (!Number.isFinite(size) || size <= 0) {
    return "Unknown size";
  }

  const units = ["B", "KB", "MB", "GB"];
  let value = size;
  let unitIndex = 0;

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
};

if (input && dropZone && fileLine) {
  const renderName = (file) => {
    fileLine.textContent = file
      ? `Selected: ${file.name}  |  ${formatBytes(file.size)}  |  ${file.type || "Unknown type"}`
      : "No file selected";
  };

  input.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    renderName(file);
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.add("active");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.remove("active");
    });
  });

  dropZone.addEventListener("drop", (event) => {
    const files = event.dataTransfer?.files;
    if (!files || files.length === 0) {
      return;
    }

    input.files = files;
    renderName(files[0]);
  });
}

if (uploadForm && submitButton) {
  uploadForm.addEventListener("submit", () => {
    submitButton.disabled = true;
    submitButton.textContent = "Analyzing...";
  });
}
