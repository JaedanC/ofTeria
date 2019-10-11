#include "Player.h"
#include "../EntityController.h"
#include "../addons/ofxDebugger/ofxDebugger.h"

Player::Player(EntityController* entityController)
	: Entity(entityController), camera(make_shared<Camera>(this))
{
	cout << "Constructing Player\n";
}

void Player::update()
{
	debugPush("Player::update()");
}

void Player::draw()
{
	debugPush("Player::draw()");
}
