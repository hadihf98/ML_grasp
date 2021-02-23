import cv2

from utils.loader import Loader
from utils.image import draw_around_box, draw_pose, get_area_of_interest
from utils.trainer import Trainer


# 1. Init loader, this will load the (non-image) dataset into memory
loader = Loader()
print(f'Dataset has {len(loader)} grasp attempts.')


# 2. Load an action and/or image
episode_index = 21380
action = loader.get_action(episode_index, action_id=0)
print(f"Loaded action of episode {action['episode_id']} with reward: {action['reward']} at pose: {action['pose']}")
# print(action)

rgbd_image = loader.get_image(episode_index, action_id=0, camera='rcd')

# You can also load action and images at once
action, rgb_image = loader.get_action(episode_index, action_id=0, images=['rc'])
cv2.imshow('image', rgb_image.mat)  # OpenCV uses uint8 format
cv2.waitKey(0)


# 3. Draw action and box on image for visualization
draw_around_box(rgbd_image, action['box_data'])
draw_pose(rgbd_image, action['pose'])
cv2.imshow('rgb image', rgbd_image.mat[:, :, :3])
cv2.imshow('depth image', rgbd_image.mat[:, :, 3])
cv2.waitKey(0)


# 4. Get image area by an affine transformation. By specifying size_result, we can scale the final image down.
area = get_area_of_interest(rgbd_image, action['pose'], size_cropped=(200, 200), size_result=(100, 100))  # [px]
cv2.imshow('area', area)
cv2.waitKey(0)


# 5. Iterate over all actions
for index, episode in enumerate(loader.yield_episodes()):
    print('Episode keys: ', list(episode.keys()))
    print('Action keys: ', list(episode['actions'][0].keys()))

    # loader.get_image(index, action_id=0, camera='rcd')
    break


# 6. Split into Training / Validation / Test set
training_set, validation_set, test_set = Trainer.split(loader.episodes, seed=42)
print(f'Training set length: {len(training_set)}')
print(f'Validation set length: {len(validation_set)}')
print(f'Test set length: {len(test_set)}')
