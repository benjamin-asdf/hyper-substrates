(ns ftlm.vehicles.art.collider
  (:require [ftlm.vehicles.art.collider-torch :as ct]))

(defprotocol CollisionHandler
  (-collide [this other state k]))

(defprotocol ColliderDetector
  (-collisions [this positions radii]))

(def impl
  (reify ColliderDetector
    (-collisions [this positions radii]
      (ct/collisions positions radii))))

(defn collisions [positions radii]
  (-collisions impl positions radii))
