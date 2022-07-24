import dataset.process
import src.generator
import src.discriminator

if __name__ == '__main__':
	dataset.process.Datasets(refresh=True)
	src.generator.Generator()
	src.discriminator.Discriminator()
