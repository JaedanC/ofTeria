#include "Player.h"
#include "../EntityController.h"

Player::Player(EntityController* entityController)
	: Entity(entityController), camera(make_shared<Camera>(this))
{
}

void Player::update()
{
}

void Player::draw()
{
}
