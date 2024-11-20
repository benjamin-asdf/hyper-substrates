(ns ftlm.disco.middlewares.time-warp
  (:require [quil.core :as q :include-macros true]))

(defn timewarp-state
  []
  {:history []
   :history-interval 500
   :history-length 10
   :last-tick nil
   :warps? false
   :warp-function rand-nth})

(defn update-time-warp
  [{:as time-warp
    :keys [last-tick history history-length warps?
           history-interval]} state]
  (if (or (not last-tick)
          (< history-interval (- (q/millis) last-tick)))
    (-> time-warp
        (assoc :last-tick (q/millis))
        (update :history conj state)
        (update :history
                (if warps?
                  identity
                  #(into [] (take-last history-length %)))))
    time-warp))

(defn time-warped
  [time-warp state]
  (if (and
       (:warps? time-warp)
       (seq (:history time-warp)))
    ((:warp-function time-warp)
     (:history time-warp))
    state))

(defn update-state
  [user-update state]
  (let [time-warp (:time-warp state)
        state (user-update state)
        time-warp (update-time-warp time-warp state)
        new-state (time-warped time-warp state)]
    (assoc new-state :time-warp time-warp)))


(defn setup-picture-jitter
  [user-setup user-settings]
  (let [initial-state (merge (timewarp-state)
                             user-settings)]
    (update-in (user-setup)
               [:time-warp]
               #(merge initial-state %))))

(defn time-warp
  [options]
  (let [user-settings (:time-warp options)
        user-update (:update options (fn [state]))
        user-setup (:setup options (fn [] {}))]
    (assoc options
      :setup #(setup-picture-jitter user-setup
                                    user-settings)
      :update (partial update-state user-update))))
