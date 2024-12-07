(ns ftlm.vehicles.art.event-q-middleware
  (:require [quil.core :as q :include-macros true]
            [ftlm.vehicles.art.lib :as lib]))

(defonce global-events-state (atom {}))

(comment
  (swap! global-events-state {}))

(defn ->events-state
  [id]
  {:sketch-id (or id
           (random-uuid)
           #_(loop [i (count global-events-state)]
               (if-not (get @global-events-state i)
                 i
                 (recur (inc i)))))
   :system-time-added #?(:clj (System/currentTimeMillis)
                         ;; correct?
                         :cljs (js/Date.now))})

(defn apply-events
  [state eventq]
  (reduce (fn [s e] (lib/event! e s)) state eventq))

;; -----------------------------------------

(defn update-state
  [user-update state]
  (if-not (-> state
              :event-q)
    (user-update state)
    (let [q (:queue (@global-events-state
                     (-> state
                         :event-q
                         :sketch-id)))
          _ (swap! global-events-state assoc-in
              [(-> state
                   :event-q
                   :sketch-id) :queue]
              [])]
      (user-update (apply-events state q)))))

;; ---------------------------------------

(defn setup
  [user-setup user-settings]
  (let [events-state (->events-state (:sketch-id
                                       user-settings))
        _ (swap! global-events-state assoc
            (:sketch-id events-state)
            events-state)]
    (update-in (user-setup)
               [:event-q]
               #(merge events-state %))))


;; ---------------------------------------

(defn on-close
  [user-on-close state]
  (when-let [id (-> state
                    :event-q
                    :sketch-id)]
    (swap! global-events-state dissoc id))
  (user-on-close state))

(defn events-middleware
  [options]
  (let [user-settings (:events-q options)
        user-update (:update options (fn [state]))
        user-setup (:setup options (fn [] {}))
        user-on-close (:on-close options (fn [state]))]
    (assoc
     options
     :setup #(setup user-setup user-settings)
     :update (partial update-state user-update)
     :on-close (partial on-close user-on-close))))

;; ---------------------------------------

(defn append-event!
  ([e] (append-event! :all e))
  ([where e]
   ;; (when
   ;;     (fn? e)
   ;;     )
   (let [append-event
           (fn [event-state]
             (update event-state :queue (fnil conj []) e))]
     (swap! global-events-state
       (fn [s]
         (case where
           :all (update-vals s append-event)
           (update s e append-event)))))))

;; ---------------------------------------

(let [track (atom {})]
  (defn append-entities-track!
    ([f] (append-event! :all f))
    ([where f]
     (append-event!
      where
      (fn [state]
        (if-let [old-ents (get track f)]
          (lib/update-ents
           state
           (fn [e]
             (if-not (old-ents (:id e))
               e
               (merge (f) e))))))))))
