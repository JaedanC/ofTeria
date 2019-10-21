#include "Entity.h"
#include "../EntityController.h"
#include "../../WorldSpawn.h"
#include "../../WorldData/WorldData.h"
#include "Player.h"
#include "Camera/Camera.h"

Entity::Entity(EntityController* entityController)
	: entityController(entityController)
{
	cout << "Constructing Entity\n";
}

void Entity::updateEntity()
{
	float bw = getEntityController()->getWorldSpawn()->getWorldData().lock()->blockWidth;
	float bh = getEntityController()->getWorldSpawn()->getWorldData().lock()->blockHeight;
	int x = bw * floor((worldPos.x + velocity.x) / bw);
	int y = bh * floor((worldPos.y + velocity.y) / bh);

	Block* block = getEntityController()->getWorldSpawn()->getWorldData().lock()->getBlock(worldPos + velocity);
	if (block) {
		getEntityController()->getPlayer().lock()->getCamera().lock()->pushCameraMatrix();
		ofSetColor(255, 0, 0);
		ofDrawRectangle(x, y, bw, bh);
		getEntityController()->getPlayer().lock()->getCamera().lock()->popCameraMatrix();
	}
	update();
}

void Entity::fixedUpdateEntity()
{
	fixedUpdate();

	worldPos += velocity;
	// TODO: Here is where you should calculate collision
	float bw = getEntityController()->getWorldSpawn()->getWorldData().lock()->blockWidth;
	float bh = getEntityController()->getWorldSpawn()->getWorldData().lock()->blockHeight;
	int x = bw * floor((worldPos.x + velocity.x) / bw);
	int y = bh * floor((worldPos.y + velocity.y) / bh);
}

void Entity::drawEntity()
{
	draw();

	// Movement Vector
	ofSetColor(ofColor::black);
	ofDrawLine(*getWorldPos(), *getWorldPos() + *getVelocity() * 10);
}
