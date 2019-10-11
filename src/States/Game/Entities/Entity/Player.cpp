#include "Player.h"
#include "ofMain.h"
#include "../EntityController.h"
#include "../addons/ofxDebugger/ofxDebugger.h"
#include "../src/Keyboard/KeyboardInput.h"
#include "../addons/ofxGameStates/ofxGameEngine.h"
#include "../src/States/PlayState.h"
#include "Camera/Camera.h"

Player::Player(EntityController* entityController)
	: Entity(entityController), camera(make_shared<Camera>(this))
{
	cout << "Constructing Player\n";
}

void Player::update()
{
	if (PlayState::Instance()->queryInput("left", QUERY_DOWN)) {
		worldPos.x -= 5;
	}
	if (PlayState::Instance()->queryInput("right", QUERY_DOWN)) {
		worldPos.x += 5;
	}
	if (PlayState::Instance()->queryInput("up", QUERY_DOWN)) {
		worldPos.y -= 5;
	}
	if (PlayState::Instance()->queryInput("down", QUERY_DOWN)) {
		worldPos.y += 5;
	}
	if (PlayState::Instance()->queryInput("zoomin", QUERY_DOWN)) {
		getCamera().lock()->getZoom() -= 0.01;
		cout << "Zoom In\n";
	}
	if (PlayState::Instance()->queryInput("zoomout", QUERY_DOWN)) {
		getCamera().lock()->getZoom() += 0.01;
		cout << "Zoom Out\n";
	}

}

void Player::draw()
{
	debugPush("Player WorldPos: " + ofToString(worldPos));
	ofSetColor(ofColor::purple);
	ofDrawRectangle(worldPos, 5, 5);
}
