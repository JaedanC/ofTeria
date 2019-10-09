#include "Player.h"
#include "../addons/ofxDebugger/ofxDebugger.h"
#include "Camera/Camera.h"

Player::Player(weak_ptr<EntityController> entityController)
	: Entity(entityController), camera(make_shared<Camera>(this))
{
}

inline weak_ptr<Camera> Player::getCamera()
{
	return camera;
}

void Player::update()
{
}

void Player::draw()
{
	debugPush("Player is writing this!");
}
