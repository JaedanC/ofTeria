#include "Camera.h"
#include "../Player.h"
#include "../../EntityController.h"
#include "../../../WorldSpawn.h"
#include "../../../WorldData/WorldData.h"

Camera::Camera(weak_ptr<Player> player)
	: player(player)
{
}

inline weak_ptr<Player> Camera::getPlayer()
{
	return player;
}

inline ofVec2f* Camera::getWorldPos()
{
	return getPlayer().lock()->getWorldPos();
}

inline void Camera::setZoom(float zoom_)
{
	zoom = zoom_;
}

void Camera::pushCameraMatrix()
{
	ofVec2f& blockDim = getPlayer().lock()->getEntityController().lock()->getWorldSpawn().lock()->getWorldData().lock()->getBlockDim();
	ofPushMatrix();
	ofScale(zoom / blockDim.x, zoom / blockDim.y);
	ofTranslate(-(*getWorldPos()));
}

void Camera::popCameraMatrix()
{
	ofPopMatrix();
}
