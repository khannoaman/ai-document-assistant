document.addEventListener("DOMContentLoaded", () => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const fileList = document.getElementById("file-list");

  function renderFileList() {
    const files = Array.from(fileInput.files);
    fileList.textContent = files.length ? files.map((f) => f.name).join(", ") : "";
  }

  if (fileInput) {
    fileInput.addEventListener("change", renderFileList);
  }

  if (dropzone) {
    ["dragover", "dragenter"].forEach((evt) =>
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
      })
    );

    ["dragleave", "drop"].forEach((evt) =>
      dropzone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
      })
    );

    dropzone.addEventListener("drop", (e) => {
      if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        renderFileList();
      }
    });
  }

  // keep the chat log scrolled to the latest message after each HTMX swap
  document.body.addEventListener("htmx:afterSwap", (e) => {
    if (e.target && e.target.id === "chat-log") {
      e.target.scrollTop = e.target.scrollHeight;
    }
  });

  // and on initial load too, in case history was restored from a prior session
  const chatLog = document.getElementById("chat-log");
  if (chatLog) {
    chatLog.scrollTop = chatLog.scrollHeight;
  }
});
