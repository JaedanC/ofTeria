#include "Player.h"
#include "../addons/ofxDebugger/ofxDebugger.h"

Player::Player(EntityController* entityController)
	: Entity(entityController), camera(this)
{
}

inline Camera* Player::getCamera()
{
	return &camera;
}

inline EntityController* Player::getEntityController()
{
	return entityController;
}

void Player::update()
{
}

void Player::draw()
{
	debugPush("Player is writing this!");
}
