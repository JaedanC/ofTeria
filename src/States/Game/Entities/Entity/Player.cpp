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
	debugPush("Velocity: " + ofToString(*getVelocity()));
	debugPush("PlayerPos: " + ofToString(worldPos));
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
	ofVec2f mouseWorldPos = getEntityController()->getWorldSpawn()->convertScreenPosToWorldPos(ofVec2f{ (float)ofGetMouseX(), (float)ofGetMouseY() });
	Block* block = getEntityController()->getWorldSpawn()->getWorldData().lock()->getBlock(mouseWorldPos);

	int blockWidth = getEntityController()->getWorldSpawn()->getWorldData().lock()->blockWidth;
	int blockHeight = getEntityController()->getWorldSpawn()->getWorldData().lock()->blockHeight;

	ofSetColor(ofColor::blue);
	if (getEntityController()->getWorldSpawn()->getWorldData().lock()->getBlock(mouseWorldPos)) {
		ofDrawRectangle(mouseWorldPos, blockWidth, blockHeight);
	}

	debugPush("Mouse WorldPos: " + ofToString(mouseWorldPos));
	// --------------------------------------------------------------------------
	// --------------------------------------------------------------------------
}
