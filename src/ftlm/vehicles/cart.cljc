(ns ftlm.vehicles.cart
  (:require
   [clojure.walk :as walk]
   [ftlm.vehicles.art.extended :as elib]
   [ftlm.vehicles.art.defs :as defs]
   [ftlm.vehicles.art.lib :as lib]
   [quil.core :as q :include-macros true]))

(defn ->rand-sensor-pair-plans
  [motor-left motor-right]
  (let [modality (rand-nth [:rays :smell :temperature])
        sensor-left-opts {:anchor :top-left
                          :modality modality
                          :shuffle-anchor? (#{:smell}
                                            modality)}
        sensor-left-opts
          (merge
            sensor-left-opts
            (when (= modality :smell)
              {:activation-shine-colors
               {:high (:misty-rose defs/color-map)
                :low (:heliotrope defs/color-map)}


               :fragrance (rand-nth [:oxygen
                                     :organic-matter])})
            (when (= modality :temperature)
              {:hot-or-cold (rand-nth [:hot :cold])}))
        sensor-right-opts (assoc sensor-left-opts
                            :anchor :top-right)
        decussates? (rand-nth [true false])
        sensor-left-id (random-uuid)
        sensor-right-id (random-uuid)
        transduction-fn (rand-nth [:excite :inhibit])]
    (case modality
      :temperature
        [[:cart/sensor sensor-left-id
          (assoc sensor-left-opts
            :anchor :middle-middle
            :activation-shine-colors
              ({:cold {:high {:h 196 :s 26 :v 100}
                       :low defs/white}
                :hot {:high (:hit-pink defs/color-map)
                      :low defs/white}}
               (:hot-or-cold sensor-left-opts)))]
         [:brain/connection :_
          {:bezier-line (lib/rand-bezier 5)
           :destination [:ref motor-left]
           :f transduction-fn
           :source [:ref sensor-left-id]}]
         [:brain/connection :_
          {:bezier-line (lib/rand-bezier 5)
           :destination [:ref motor-right]
           :f transduction-fn
           :source [:ref sensor-left-id]}]]
      [[:cart/sensor sensor-left-id sensor-left-opts]
       [:cart/sensor sensor-right-id sensor-right-opts]
       [:brain/connection :_
        {:bezier-line (lib/rand-bezier 5)
         :destination [:ref motor-left]
         :f transduction-fn
         :source [:ref
                  (if decussates?
                    sensor-right-id
                    sensor-left-id)]}]
       [:brain/connection :_
        {:bezier-line (lib/rand-bezier 5)
         :destination [:ref motor-right]
         :f transduction-fn
         :source [:ref
                  (if decussates?
                    sensor-left-id
                    sensor-right-id)]}]])))

(defn ->love-wires
  [motor-left motor-right sensor-opts]
  (let [sensor-left-opts (merge sensor-opts {:anchor :top-left})
        sensor-right-opts (assoc sensor-left-opts :anchor :top-right)
        sensor-left-id (random-uuid)
        sensor-right-id (random-uuid)
        decussates? false]
    [[:cart/sensor sensor-left-id sensor-left-opts]
     [:cart/sensor sensor-right-id sensor-right-opts]
     [:brain/connection :_
      {:destination [:ref motor-left]
       :f :inhibit
       :source [:ref (if decussates? sensor-right-id sensor-left-id)]}]
     [:brain/connection :_
      {:destination [:ref motor-right]
       :f :inhibit
       :source [:ref (if decussates? sensor-left-id sensor-right-id)]}]]))

(defn random-multi-sensory
  [sensor-pair-count]
  (fn [{:as opts :keys [baseline-arousal]}]
    {:body (merge opts
                  {:color-of-the-mind
                     (rand-nth [:cyan :hit-pink
                                :navajo-white :sweet-pink
                                :woodsmoke :mint
                                :midnight-purple])})
     :components
       (into [[:cart/motor :motor-left
               {:activation-shine-colors
                  {:high (:misty-rose defs/color-map)}
                :activation-shine-speed 0.5
                :anchor :bottom-left
                :corner-r 5
                :on-update [(lib/->cap-activation)]
                :rotational-power 0.02}]
              [:cart/motor :motor-right
               {:activation-shine-colors
                  {:high (:misty-rose defs/color-map)}
                :activation-shine-speed 0.5
                :anchor :bottom-right
                :corner-r 5
                :on-update [(lib/->cap-activation)]
                :rotational-power 0.02}]
              [:brain/neuron :arousal
               {:activation-shine true
                :activation-shine-colors
                  {:high (:red defs/color-map)}
                :nucleus :arousal
                :on-update [(lib/->baseline-arousal
                              (or baseline-arousal 0.8))]}]
              [:brain/connection :_
               {:destination [:ref :motor-left]
                :f rand
                :hidden? true
                :source [:ref :arousal]}]
              [:brain/connection :_
               {:destination [:ref :motor-right]
                :f rand
                :hidden? true
                :source [:ref :arousal]}]]
             ;; (->love-wires :motor-left :motor-right
             ;; {:modality :smell
             ;; :fragrance :oxygen})
             (mapcat identity
               (repeatedly sensor-pair-count
                           (fn []
                             (->rand-sensor-pair-plans
                               :motor-right
                               :motor-left)))))}))

(def body-plans
  {:multi-sensory (random-multi-sensory 6)})

(defn shuffle-anchor [{:keys [shuffle-anchor?] :as e}]
  (if-not shuffle-anchor?
    e
    (let [[x y] (lib/anchor->trans-matrix (:anchor e))
          anch-pos
          [(lib/normal-distr x 0.2)
           (lib/normal-distr y 0.12)]]
      (assoc e :anchor-position anch-pos))))

(def builders
  {:brain/connection
   (comp
    lib/->connection
    #(walk/prewalk-replace
      {:excite lib/excite
       :inhibit lib/inhibit}
      %))
   :brain/neuron
   (comp elib/with-electrode-sensitivity lib/->neuron)
   :cart/body
   (fn [opts]
     (lib/->body
      (merge
       {:color (:sweet-pink defs/color-map)
        :corner-r 10
        :darts? true
        :draggable? true
        :on-update-map
        {:indicator
         (lib/every-n-seconds
          1
          (fn [e s _]
            (if (= (:id e) (:id (:selection s)))
              (assoc e
                     :stroke-weight 4
                     :stroke (:amethyst-smoke
                              defs/color-map))
              (dissoc e :stroke-weight :stroke))))}
        :pos (lib/rand-on-canvas-gauss 0.3)
        :rot (* (rand) q/TWO-PI)
        :scale 1}
       opts)))
   :cart/motor (comp elib/with-electrode-sensitivity lib/->motor)
   :cart/sensor
   (comp
    ;; elib/with-electrode-sensitivity
    shuffle-anchor
    lib/->sensor)})

(defmulti build-entity first)

(defmethod build-entity :cart/entity [[_ {:keys [f]}]] (f))

(defmethod build-entity :default [[kind opts]] ((builders kind) opts))

(defn ref? [v] (and (sequential? v) (= (first v) :ref)))

;; only have maps 1 deep right now

(defn resolve-refs
  [temp-id->ent form]
  (update-vals
   form
   (fn [v]
     (if (ref? v)
       (or (temp-id->ent (second v))
           (throw #?(:cljs (throw (js/Error.
                                   (str
                                    (second v)
                                    " is not resolved")))
                     :clj (Exception.
                           (str (second v)
                                " is not resolved")))))
       v))))

(defn ->cart
  [{:keys [body components]}]
  (let [body (build-entity [:cart/body body])
        {:keys [comps]}
        (reduce (fn [{:keys [comps temp-id->ent]}
                     [kind temp-id opts]]
                  (let [entity (build-entity
                                [kind
                                 (resolve-refs
                                  temp-id->ent
                                  opts)])]
                    {:comps (into comps
                                  (if (map? entity)
                                    [entity]
                                    entity))
                     :temp-id->ent (if (= temp-id :_)
                                     temp-id->ent
                                     (assoc temp-id->ent
                                            temp-id entity))}))
                {:comps [] :temp-id->ent {}}
                components)]
    (into [(assoc body
                  :components (into [] (map :id) comps))]
          comps)))






#_(defn some-rand-environment-things
    [defs n]
    (let [stuff (repeatedly n
                            #(rand-nth [:temp-cold :temp-hot
                                        :organic-matter
                                        :oxygen]))
          ->make
          {:organic-matter
           (fn []
             (elib/->organic-matter
              {:odor {:decay-rate 2 :intensity 40}
               :pos (lib/rand-on-canvas-gauss 0.5)}))
           :oxygen (fn []
                     (elib/->oxygen
                      {:odor {:decay-rate 2 :intensity 40}
                       :pos (lib/rand-on-canvas-gauss
                             0.2)}))
           :temp-cold (fn []
                        (elib/->temperature-bubble-1
                         (rand-temperature-bubble defs :cold)))
           :temp-hot (fn []
                       (elib/->temperature-bubble-1
                        (rand-temperature-bubble defs
                                                 :hot)))}]
      (mapcat (fn [op] (op)) (map ->make stuff))))







;; (->cart plan)
