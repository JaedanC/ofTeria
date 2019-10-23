#include "Entity.h"
#include "../EntityController.h"
#include "../../WorldSpawn.h"
#include "../../WorldData/WorldData.h"
#include "../../../PlayState.h"
#include "Player.h"
#include "Camera/Camera.h"


ofVec2f rectDist(ofVec2f& b1, ofVec2f& b1dim, ofVec2f& b2, ofVec2f& b2dim) {
	ofVec2f dist;
	dist.x = abs(b1.x + b1dim.x / 2.0f - b2.x - b2dim.x / 2.0f) - abs(b1dim.x + b2dim.x);
	dist.y = abs(b1.y + b1dim.y / 2.0f - b2.y - b2dim.y / 2.0f) - abs(b1dim.y + b2dim.y);
	return dist;
};


Entity::Entity(EntityController* entityController, bool affectedByGravity)
	: entityController(entityController), affectedByGravity(affectedByGravity)
{
	cout << "Constructing Entity\n";
}

void Entity::updateEntity()
{
	update();
	interp = *getVelocity() * PlayState::Instance()->framePercentage;
}

void Entity::calculateCollision()
{
	vector<Block*> localBlocks;
	ofVec2f newLocationTL = *getPrevWorldPos() + velocity + getHitbox()->getOffset();
	ofVec2f newLocationBR = newLocationTL + getHitbox()->getHitboxDimension();
	ofVec2f playerCentre = newLocationTL / 2.0f + newLocationTL / 2.0f;

	ofRectangle r(newLocationTL, newLocationBR);


	glm::uvec2& blockDim = getEntityController()->getWorldSpawn()->getWorldData().lock()->getBlockDim();

	getEntityController()->getPlayer().lock()->getCamera().lock()->pushCameraMatrix();
	for (int x = floor(newLocationTL.x); x < ceil(newLocationBR.x); x += blockDim.x) {
		for (int y = floor(newLocationTL.y); y < ceil(newLocationBR.y); y += blockDim.y) {
			Block * block = getEntityController()->getWorldSpawn()->getWorldData().lock()->getBlock(ofVec2f(x, y));
			if (block) {
				localBlocks.push_back(block);
			}

			ofSetColor(ofColor::red);
			ofDrawRectangle(r);

		}
	}

	getEntityController()->getPlayer().lock()->getCamera().lock()->popCameraMatrix();

	//worldPos.x += velocity.x - maxIntersection.getWidth() * ofSign(velocity.x);
	//worldPos.y += velocity.y - maxIntersection.getHeight() * ofSign(velocity.y);

	worldPos += velocity;
}

void Entity::fixedUpdateEntity()
{
	// Gravity
	//addVelocity(ofVec2f{ 0, 0.1 });

	calculateCollision();

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
