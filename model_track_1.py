from utils import *
from sklearn.model_selection import train_test_split

print('Setting up...')

#path for track 1
path = 'data/track_2'

#path for track 2
# path = 'data/track_2'

data = importDataInfo(path)

balanceData(data, display=False)
# visualizeData(data)

imagesPath, steering = loadData(path, data)

xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steering, test_size=0.2, random_state=10)
print('Total Training Images: ', len(xTrain))
print('Total Validation Images: ', len(xVal))

model = buildModelTrack1()
model.summary()

history = model.fit(batchGen(xTrain, yTrain, 100, 1), steps_per_epoch=300, epochs=10,
        validation_data=batchGen(xVal, yVal, 100, 0), validation_steps=200)

model.save('./models/model_self_track_1.h5') #for track 1
# model.save('./models/model_self_track_2.h5') #for track 2
print('model saved')

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.legend(['training', 'validation'])
# plt.ylim([0, 1])
# plt.title("Loss")
# plt.xlabel('epoch')
# plt.show()