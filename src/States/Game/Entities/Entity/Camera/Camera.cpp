#include "Camera.h"
#include "../Player.h"
#include "../../EntityController.h"
#include "../../../WorldSpawn.h"

Camera::Camera(Player* player)
	: player(player)
{
}

inline Player* Camera::getPlayer()
{
	return player;
}

inline ofVec2f* Camera::getWorldPos()
{
	return getPlayer()->getWorldPos();
}

inline void Camera::setZoom(float zoom_)
{
	zoom = zoom_;
}

void Camera::pushCameraMatrix()
{
	ofVec2f& blockDim = getPlayer()->getEntityController()->getWorldSpawn()->getWorldData()->getBlockDim();
	ofPushMatrix();
	ofScale(zoom / blockDim.x, zoom / blockDim.y);
	ofTranslate(-(*getWorldPos()));
}

void Camera::popCameraMatrix()
{
	ofPopMatrix();
}
