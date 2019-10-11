#include "Camera.h"
#include "ofMain.h"
#include "../Player.h"
#include "../Entity.h"
#include "../../EntityController.h"
#include "../../../WorldSpawn.h"
#include "../../../WorldData/WorldData.h"

Camera::Camera(Player* player)
	: player(player)
{
	cout << "Constructing Camera\n";
}

ofVec2f* Camera::getPlayerPos()
{
	return getPlayer()->getWorldPos();
}

void Camera::pushCameraMatrix()
{

	ofVec2f& blockDim = getPlayer()->getEntityController()->getWorldSpawn()->getWorldData().lock()->getBlockDim();
	ofPushMatrix();
	ofScale(zoom / blockDim.x, zoom / blockDim.y);
	ofTranslate(-(*getPlayerPos()));
}

void Camera::popCameraMatrix()
{
	ofPopMatrix();
}
