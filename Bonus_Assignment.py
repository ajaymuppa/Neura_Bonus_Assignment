#Question 1: Question Answering with Transformers

#Use Hugging Face’s transformers library to build a simple question answering system using pre-trained models.
#Setup Instructions:
#Before starting, make sure your Python environment has the transformers and torch libraries installed.

#Assignment Tasks:

#1. Basic Pipeline Setup
#•	Import the pipeline function from transformers.
#•	Initialize a question-answering pipeline using the default model.
#•	Ask a question based on the given context.
#Expected output
#•	'answer': 'Charles Babbage' (or close variant)
#•	A confidence 'score' key with a float value above 0.65
#•	Valid 'start' and 'end' indices

#CODE:
from transformers import pipeline

qa_pipeline = pipeline("question-answering")

context = """Charles Babbage was an English polymath. A mathematician, philosopher, inventor and mechanical engineer, 
Babbage originated the concept of a digital programmable computer."""

question = "Who is known as the father of the computer?"

result = qa_pipeline(question=question, context=context)
print(result)


#2. Use a Custom Pretrained Model
#•	Switch to a different QA model like deepset/roberta-base-squad2.
#Expected output
#•	'answer': 'Charles Babbage'
#•	'score' greater than 0.70
#•	Include 'start' and 'end' indices

#CODE:
from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

context = """Charles Babbage was an English polymath. A mathematician, philosopher, inventor and mechanical engineer, 
Babbage originated the concept of a digital programmable computer."""

question = "Who is considered the father of computing?"

result = qa_pipeline(question=question, context=context)
print(result)


#3. Test on Your Own Example
#•	Write your own 2–3 sentence context.
#•	Ask two different questions from it and print the answers.
#Expected output
#•	Include a relevant, meaningful 'answer' to each question
#•	Display a 'score' above 0.70 for each answer

#CODE:
context = "The Eiffel Tower is located in Paris, France. It was completed in 1889 and remains one of the most visited monuments in the world."

questions = [
    "Where is the Eiffel Tower?",
    "When was the Eiffel Tower completed?"
]

for q in questions:
    result = qa_pipeline(question=q, context=context)
    print(f"Q: {q}\nA: {result['answer']} (Score: {result['score']:.2f})\n")



#Question2: 
#1.	Digit-Class Controlled Image Generation with Conditional GAN
#Objective: 
#Implement a Conditional GAN that generates MNIST digits based on a given class label (0–9). The goal is to understand how conditioning GANs on labels affects generation and how class control is added.
#Task Description
#1.	Modify a basic GAN to accept a digit label as input.
#2.	Concatenate the label embedding with both:
#o	the noise vector (input to Generator),
#o	the image input (to the Discriminator).
#3.	Train the cGAN on MNIST and generate digits conditioned on specific labels (e.g., generate only 3s or 7s).
#4.	Visualize generated digits label by label (e.g., one row per digit class).
#Expected Output
#•	A row of 10 generated digits, each conditioned on labels 0 through 9.
#•	Generator should learn to control output based on class.
#•	Loss curves may still fluctuate, but quality and label accuracy improves over time.

#CODE:
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_dim = 100
num_classes = 10
embedding_dim = 50
image_size = 28 * 28
batch_size = 128
lr = 0.0002
epochs = 100

# Data loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_loader = DataLoader(
    torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True),
    batch_size=batch_size, shuffle=True
)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embedding_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + embedding_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, image_size),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat((noise, self.label_embed(labels)), dim=1)
        img = self.model(x)
        return img.view(-1, 1, 28, 28)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embedding_dim)
        self.model = nn.Sequential(
            nn.Linear(image_size + embedding_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        x = torch.cat((img_flat, self.label_embed(labels)), dim=1)
        return self.model(x)

# Initialize
G = Generator().to(device)
D = Discriminator().to(device)

# Optimizers and loss
loss_fn = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# Create directory for samples
os.makedirs("samples", exist_ok=True)

# Training
G_losses, D_losses = [], []
for epoch in range(epochs):
    for i, (real_imgs, labels) in enumerate(train_loader):
        batch = real_imgs.size(0)
        real_imgs, labels = real_imgs.to(device), labels.to(device)

        # Adversarial ground truths
        real = torch.ones(batch, 1).to(device)
        fake = torch.zeros(batch, 1).to(device)

        # -----------------
        # Train Generator
        # -----------------
        z = torch.randn(batch, latent_dim).to(device)
        gen_labels = torch.randint(0, num_classes, (batch,)).to(device)
        gen_imgs = G(z, gen_labels)
        validity = D(gen_imgs, gen_labels)
        g_loss = loss_fn(validity, real)

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

        # ---------------------
        # Train Discriminator
        # ---------------------
        real_validity = D(real_imgs, labels)
        d_real_loss = loss_fn(real_validity, real)

        fake_validity = D(gen_imgs.detach(), gen_labels)
        d_fake_loss = loss_fn(fake_validity, fake)

        d_loss = d_real_loss + d_fake_loss

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

    G_losses.append(g_loss.item())
    D_losses.append(d_loss.item())

    print(f"[Epoch {epoch+1}/{epochs}] D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

    # Save sample
    if (epoch + 1) % 10 == 0 or epoch == 0:
        with torch.no_grad():
            fixed_noise = torch.randn(10, latent_dim).to(device)
            fixed_labels = torch.arange(0, 10).to(device)
            generated = G(fixed_noise, fixed_labels)
            save_image(make_grid(generated, nrow=10, normalize=True), f"samples/epoch_{epoch+1}.png")

# Plot losses
plt.plot(G_losses, label="Generator Loss")
plt.plot(D_losses, label="Discriminator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Losses")
plt.savefig("samples/loss_plot.png")
plt.show()



#Short Answer
#•	How does a Conditional GAN differ from a vanilla GAN?
#→ Include at least one real-world application where conditioning is important.
#Answer: A vanilla GAN generates data with no label control. A Conditional GAN (cGAN) allows class-specific control by feeding class labels to both the generator and discriminator.

#Example Use Case:

#In facial synthesis, generate faces by gender, age, or emotion (e.g., generate a smiling 30-year-old female).



#•	What does the discriminator learn in an image-to-image GAN?
#→ Why is pairing important in this context?
#Answer: The discriminator in an image-to-image GAN (e.g., Pix2Pix) learns whether the generated image is a realistic transformation of a specific input.

#Pairing is crucial because:

#Without matching input-output pairs, the model may produce plausible but unrelated outputs (e.g., a cat for a dog sketch).

#It ensures semantic and structural fidelity between the source and generated domains.

