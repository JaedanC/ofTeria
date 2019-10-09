#include "Entity.h"

Entity::Entity(weak_ptr<EntityController> entityController)
	: entityController(entityController)
{
}

inline ofVec2f* Entity::getWorldPos()
{
	return &worldPos;
}

inline weak_ptr<EntityController> Entity::getEntityController()
{
	return entityController;
}
