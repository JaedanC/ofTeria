#include "Camera.h"
#include "ofMain.h"
#include "../Player.h"
#include "../Entity.h"
#include "../../EntityController.h"
#include "../../../WorldSpawn.h"
#include "../../../WorldData/WorldData.h"

Camera::Camera(Player* player)
	: player(player->weak_from_this())
{
}

ofVec2f* Camera::getPlayerPos()
{
	return getPlayer().lock()->getWorldPos();
}

void Camera::pushCameraMatrix()
{

	ofVec2f& blockDim = getPlayer().lock()->getEntityController().lock()->getWorldSpawn().lock()->getWorldData().lock()->getBlockDim();
	ofPushMatrix();
	ofScale(zoom / blockDim.x, zoom / blockDim.y);
	ofTranslate(-(*getPlayerPos()));
}

void Camera::popCameraMatrix()
{
	ofPopMatrix();
}
