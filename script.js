document.getElementById("searchButton").addEventListener("click", async () => {
  const query = document.getElementById("searchInput").value;

  if (!query) {
    alert("Please enter a search query!");
    return;
  }

  const backendUrl = "http://127.0.0.1:5000/search";

  try {
    // Make a POST request to the backend
    const response = await fetch(backendUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    });
    console.log("Here");
    console.log(response);

    if (!response.ok) {
      throw new Error("Error fetching results!");
    }

    const passages = await response.json();
    console.log(passages);
    await displayPassages(passages);
  } catch (error) {
    console.error("Error:", error.message);
    alert("Failed to fetch results. Displaying dummy data instead.");
    displayDummyData();
  }
});

async function displayPassages(passages) {
  const resultsContainer = document.getElementById("resultsContainer");
  resultsContainer.innerHTML = ""; // Clear previous results

  if (passages.length === 0) {
    resultsContainer.innerHTML = "<p>No results found.</p>";
    return;
  }

  passages.forEach((passage) => {
    const passageDiv = document.createElement("div");
    passageDiv.className = "result-item";

    const passageId = document.createElement("h3");
    passageId.textContent = `Passage ID: ${passage.id}`;

    const passageText = document.createElement("p");
    passageText.textContent = passage.passage;

    passageDiv.appendChild(passageId);
    passageDiv.appendChild(passageText);
    resultsContainer.appendChild(passageDiv);
  });
}

function displayDummyData() {
  // Dummy data to display in case of an error
  const dummyData = [
    { id: "0", passage: "This is a dummy passage. Backend is unreachable." },
    { id: "1", passage: "Here is another dummy passage for your query." },
    { id: "2", passage: "Dummy data helps simulate results during failures." },
  ];

  displayPassages(dummyData);
}
