#include "Player.h"
#include "ofMain.h"
#include "../EntityController.h"
#include "../addons/ofxDebugger/ofxDebugger.h"
#include "../src/Keyboard/KeyboardInput.h"
#include "../addons/ofxGameStates/ofxGameEngine.h"
#include "../src/States/PlayState.h"
#include "Camera/Camera.h"
#include "../EntityController.h"
#include "../../WorldSpawn.h"
#include "../../WorldData/WorldData.h"

Player::Player(EntityController* entityController)
	: Entity(entityController), camera(make_shared<Camera>(this))

{
	cout << "Constructing Player\n";
	//25, 37 size hitbox
	float w = 25; float h = 37;
	hitbox.set(getWorldPos(), -ofVec2f{w / 2, h / 2}, w, h);
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

	// Friction
	//*getVelocity() = *getVelocity() * pow(0.9, ofGetLastFrameTime() * 60);

	// Gravity
	// TODO
}

void Player::draw()
{
	debugPush("Player WorldPos: " + ofToString(worldPos));
	getHitbox()->draw();

	ofVec2f check_location = getEntityController()->getWorldSpawn()->convertScreenPosToWorldPos(ofVec2f{ (float)ofGetMouseX(), (float)ofGetMouseY() });
	Block* block = getEntityController()->getWorldSpawn()->getWorldData().lock()->getBlock(check_location);

	auto camera = getCamera().lock();
	int blockWidth = getEntityController()->getWorldSpawn()->getWorldData().lock()->blockWidth;
	int blockHeight = getEntityController()->getWorldSpawn()->getWorldData().lock()->blockHeight;

	debugPush("Mouse WorldPos: " + ofToString(check_location));

	ofSetColor(ofColor::blue);
	if (getEntityController()->getWorldSpawn()->getWorldData().lock()->getBlock(check_location)) {
		ofDrawRectangle(check_location, blockWidth, blockHeight);
	}
}
