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
	return getPlayer()->getPrevWorldPos();
}

void Camera::pushCameraMatrix()
{
	debugPush("Zoom Level: " + ofToString(zoom));
	glm::uvec2& blockDim = getPlayer()->getEntityController()->getWorldSpawn()->getWorldData().lock()->getBlockDim();
	ofPushMatrix();
	ofScale(1 / zoom);

	/* This is the offset of the camera with respect to the size of the screen and
	the zoom factor. */
	offsetPos = zoom * ofVec2f(ofGetWidth(), ofGetHeight()) / 2.0;

	/* This is the interpolation of the player's movement. This ensures that all other entities do not
	need to worry about the player's interpolation as this is being handled by the camera.
	Essentially, naturally the camera wants to be jerky as it follows the exact x, y of the player
	but due to fixedUpdate not being in sync we interpolate between the currentPos of the player
	and the previousPos of the player between every fixedUpdate() to make everything still look smooth.
	However everything in between is fake and not actually calculated. */
	ofVec2f playerInterpolation = PlayState::Instance()->framePercentage * *getPlayer()->getVelocity();

	/* Finally, apply all transformations together.*/
	ofTranslate(offsetPos - *getPlayerPos() - playerInterpolation);
}

void Camera::popCameraMatrix()
{
	ofPopMatrix();
}
