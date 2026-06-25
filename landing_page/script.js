const form = document.getElementById('contact-form');
if (form) {
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    const name = document.getElementById('name')?.value.trim();
    const email = document.getElementById('email')?.value.trim();
    if (!name || !email) {
      alert('Please provide both your name and email.');
      return;
    }
    alert(`Thanks, ${name}! We will contact you at ${email} shortly.`);
    form.reset();
  });
}
