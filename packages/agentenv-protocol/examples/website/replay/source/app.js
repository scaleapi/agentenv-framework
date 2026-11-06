document.getElementById("subscribe").addEventListener("submit", async (event) => {
  event.preventDefault();
  const email = new FormData(event.target).get("email");
  const answer = await fetch("api/subscribe", {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({email}),
  });
  document.getElementById("subscribed").textContent = answer.ok ? `Subscribed ${email}` : "Try again";
});
