#include "Player.h"
#include "ofMain.h"
#include "Camera/Camera.h"
#include "../addons/ofxDebugger/ofxDebugger.h"
#include "../addons/ofxGameStates/ofxGameEngine.h"
#include "../EntityController.h"
#include "../src/Keyboard/KeyboardInput.h"
#include "../src/States/PlayState.h"
#include "../../WorldSpawn.h"
#include "../../WorldData/WorldData.h"

Player::Player(EntityController* entityController)
	: Entity(entityController), camera(make_shared<Camera>(this))

{
	cout << "Constructing Player\n";
	hitbox.setOffset(this, -ofVec2f{ width / 2, height / 2}, width, height);
}

void Player::update()
{
	debugPush("GetLastFrameTime: " + ofToString(ofGetLastFrameTime()));
	debugPush("Velocity: " + ofToString(*getVelocity()));
}

void Player::fixedUpdate()
{
	if (PlayState::Instance()->queryInput("left", QUERY_DOWN)) {
		getVelocity()->x -= 1;
	}
	if (PlayState::Instance()->queryInput("right", QUERY_DOWN)) {
		getVelocity()->x += 1;
	}
	if (PlayState::Instance()->queryInput("up", QUERY_DOWN)) {
		getVelocity()->y -= 1;
	}
	if (PlayState::Instance()->queryInput("down", QUERY_DOWN)) {
		getVelocity()->y += 1;
	}
	if (PlayState::Instance()->queryInput("zoomin", QUERY_DOWN)) {
		getCamera().lock()->setZoom(getCamera().lock()->getZoom() - 0.01);
	}
	if (PlayState::Instance()->queryInput("zoomout", QUERY_DOWN)) {
		getCamera().lock()->setZoom(getCamera().lock()->getZoom() + 0.01);
	}

	// TODO
	// Friction
	// Gravity
}

void Player::draw()
{

	// --------------------------------------------------------------------------
	// --------------------------------------------------------------------------
	// Testing the getBlock function
	ofVec2f check_location = getEntityController()->getWorldSpawn()->convertScreenPosToWorldPos(ofVec2f{ (float)ofGetMouseX(), (float)ofGetMouseY() });
	Block* block = getEntityController()->getWorldSpawn()->getWorldData().lock()->getBlock(check_location);

	int blockWidth = getEntityController()->getWorldSpawn()->getWorldData().lock()->blockWidth;
	int blockHeight = getEntityController()->getWorldSpawn()->getWorldData().lock()->blockHeight;

	debugPush("Player WorldPos: " + ofToString(worldPos));
	debugPush("Mouse WorldPos: " + ofToString(check_location));

	ofSetColor(ofColor::blue);
	if (getEntityController()->getWorldSpawn()->getWorldData().lock()->getBlock(check_location)) {
		ofDrawRectangle(check_location, blockWidth, blockHeight);
	}
	// --------------------------------------------------------------------------
	// --------------------------------------------------------------------------
}
