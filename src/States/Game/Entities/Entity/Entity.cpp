#include "Entity.h"
#include "../EntityController.h"
#include "../../WorldSpawn.h"
#include "../../WorldData/WorldData.h"
#include "../../../PlayState.h"
#include "Player.h"
#include "Camera/Camera.h"

Entity::Entity(EntityController* entityController)
	: entityController(entityController)
{
	cout << "Constructing Entity\n";
}

void Entity::updateEntity()
{
	update();
	interp = *getVelocity() * PlayState::Instance()->framePercentage;
}

void Entity::fixedUpdateEntity()
{
	worldPos += velocity;

	// TODO: Here is where you should calculate collision
	//float bw = getEntityController()->getWorldSpawn()->getWorldData().lock()->blockWidth;
	//float bh = getEntityController()->getWorldSpawn()->getWorldData().lock()->blockHeight;
	//int x = bw * floor((worldPos.x + velocity.x) / bw);
	//int y = bh * floor((worldPos.y + velocity.y) / bh);

	fixedUpdate();

	prevWorldPos = worldPos;
}

void Entity::drawEntity()
{
	getHitbox()->draw();
	draw();

	// Movement Vector
	ofSetColor(ofColor::black);
	ofDrawLine(*getPrevWorldPos() + *getInterp(), *getPrevWorldPos() + *getVelocity() * 10 + *getInterp());
}
