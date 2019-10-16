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
	//25, 37 size hitbox
	float w = 25; float h = 37;
	hitbox.set(getWorldPos(), -ofVec2f{w / 2, h / 2}, w, h);
}

void Player::update()
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
		getCamera().lock()->getZoom() -= 0.01;
	}
	if (PlayState::Instance()->queryInput("zoomout", QUERY_DOWN)) {
		getCamera().lock()->getZoom() += 0.01;
	}

	// Friction
	*getVelocity() *= 0.99;

	// Gravity
	// TODO
}

void Player::draw()
{
	debugPush("Player WorldPos: " + ofToString(worldPos));
	getHitbox()->draw();
}
