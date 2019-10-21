#include "Camera.h"
#include "ofMain.h"
#include "../Player.h"
#include "../Entity.h"
#include "../../EntityController.h"
#include "../../../WorldSpawn.h"
#include "../../../WorldData/WorldData.h"
#include "../addons/ofxDebugger/ofxDebugger.h"
#include "../../../../PlayState.h"

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
	debugPush("Zoom Level: " + ofToString(zoom));
	glm::uvec2& blockDim = getPlayer()->getEntityController()->getWorldSpawn()->getWorldData().lock()->getBlockDim();
	ofPushMatrix();
	ofScale(1 / zoom);

	offsetPos = zoom * ofVec2f(ofGetWidth(), ofGetHeight()) / 2.0;
	ofTranslate(-(*getPlayerPos() - offsetPos));
}

void Camera::popCameraMatrix()
{
	ofPopMatrix();
}
