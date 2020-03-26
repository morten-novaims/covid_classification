### Log the results in Neptune.ML

import neptune

# define parameters
PARAMS = {"learning_rate": 0.0015,
		   "batch_size": 16}

# The init() function called this way assumes that 
# NEPTUNE_API_TOKEN environment variable is defined.

neptune.init('morten/covid-classification',
	      	api_token=YOUR_API_TOKEN)
neptune.create_experiment(name='trial_example',
			  			  params=PARAMS)

# log some metrics
neptune.log_metric('loss', 6)
neptune.log_metric('val_loss', 7)
neptune.log_metric('val_acc', 8)
neptune.log_metric('acc', 9)

import matplotlib.pyplot as plt

plt.plot(range(100))
plt.savefig('trial.png')
plt.show()


# log the image
neptune.log_image('TRIAL Graph', 'trial.png')
