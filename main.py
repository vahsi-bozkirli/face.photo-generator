import preprocess
import model
import training
import generate
import matplotlib.pyplot as plt

data_loader = preprocess.read_preprocess()
autoencoder = model.Autoencoder()
autoencoder = training.train_model(data_loader=data_loader, model=autoencoder)
data_loader = preprocess.read("data/test/")
dataiter = iter(data_loader)
image, _ = next(dataiter)

if __name__ == "__main__":
    output = generate.predict(autoencoder,image)
    plt.imshow(output[0].permute(1, 2, 0).numpy());
    plt.axis('off');
    plt.show();

