from sklearn.model_selection import train_test_split
from utils import *

# Dataset
Folder_path = "./data/recovery_data/"
Img_path = Folder_path + "IMG/"
df = pd.read_csv(Folder_path + "driving_log.csv",
                 names=['center', 'left', 'right', 'steering', 'gas', 'brake', 'speed'])

center = df.center.tolist()
left = df.left.tolist()
right = df.right.tolist()
steering = df.steering.tolist()
leftSteering = (df.steering + 0.25).tolist()
rightSteering = (df.steering - 0.25).tolist()

X_train = center + left + right
y_train = steering + leftSteering + rightSteering

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)
X_trf, y_trf = getTransformedImages(X_train, y_train, n_each=10)

X_trf_val, y_trf_val = getTransformedImages(X_valid, y_valid, n_each=1)
X_train, y_train, X_valid, y_valid = X_trf, y_trf, X_trf_val, y_trf_val

model = buildModelTrack2()
model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, shuffle=True,
          validation_data=(X_valid, y_valid))
model.save('./models/model_self_track_2_final.h5')

# Save as json for inspection of architecture
with open('model_final.json', 'w') as file:
    file.write(model.to_json())